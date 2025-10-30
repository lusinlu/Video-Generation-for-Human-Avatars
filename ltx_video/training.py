import os
import argparse
import json
from lpips import LPIPS
from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from ltx_video.utils.torch_utils import (
    save_training_checkpoint,
    save_module_safetensors,
)
from ltx_video.config import TrainConfig, load_train_config_from_yaml
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import (
    Transformer3DModel,
    AudioProjection,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.dataset import LatentPairDataset, ValidationDataset, collate_latent_pairs
from ltx_video.validation import validate_epoch, validate_video
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    vae_decode as _vae_decode,
    normalize_latents as _norm,
)


class CombinedTrainModel(torch.nn.Module):
    """
    Wraps the transformer and optional decoder_vae into a single module so DeepSpeed
    manages parameters for both. The forward proxies to the transformer.
    """

    def __init__(
        self,
        transformer: Transformer3DModel,
        decoder_vae: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.transformer = transformer
        if decoder_vae is not None:
            self.decoder_vae = decoder_vae
        else:
            self.decoder_vae = None

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


try:
    import deepspeed
except ImportError:
    raise ImportError(
        "DeepSpeed not installed! Install with: pip install deepspeed\n"
        "Or set use_deepspeed: false in your config."
    )


def build_transformer(config: TrainConfig):
    ltxv_model_path = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=config.checkpoint_path,
        repo_type="model",
    )
    if config.precision == "bfloat16":
        model = Transformer3DModel.from_pretrained(ltxv_model_path).to(torch.bfloat16)
    else:
        model = Transformer3DModel.from_pretrained(ltxv_model_path)

    model.config.caption_channels = int(config.audio_embed_dim)

    model.caption_projection = AudioProjection(
        in_features=config.audio_embed_dim, hidden_size=model.inner_dim
    )

    if config.precision == "bfloat16":
        model.caption_projection = model.caption_projection.to(torch.bfloat16)

    return model


def apply_training_strategy(
    model: torch.nn.Module, config: TrainConfig, train_mode: str
):
    """
    Configure trainable params:
    - train_mode == "full": train entire model (no LoRA)
    - train_mode == "lora_audio": apply LoRA to cross-attn and train LoRA + caption_projection
    """
    if train_mode == "lora_audio":
        target_modules = []
        for i in range(len(model.transformer_blocks)):
            target_modules.extend(
                [
                    f"transformer_blocks.{i}.attn2.to_q",
                    f"transformer_blocks.{i}.attn2.to_k",
                    f"transformer_blocks.{i}.attn2.to_v",
                    f"transformer_blocks.{i}.attn2.to_out.0",
                ]
            )
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        for n, p in model.named_parameters():
            if ("lora_" in n) or ("caption_projection" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model
    else:
        for _n, p in model.named_parameters():
            p.requires_grad = True
        return model


def build_velocity_target(
    tokens: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    dt_eps: float = 1e-4,
):
    """
    tokens: x0 (clean), shape [B, seq_len, dim]
    noise: eps, same shape
    t: [B]
    scheduler: RectifiedFlowScheduler
    Returns: x_t, v_target
    """
    alpha = scheduler.alpha(t)
    sigma = scheduler.sigma(t)
    # broadcast [B] -> [B, 1, 1, ...] matching tokens
    while alpha.dim() < tokens.dim():
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

    # forward noising
    x_t = alpha * tokens + sigma * noise

    # reconstruct x0
    x0_recon = (x_t - sigma * noise) / (alpha + 1e-12)

    # derivatives: alpha'(t), sigma'(t)
    a_dot = scheduler.alpha_dot(t)
    s_dot = scheduler.sigma_dot(t)
    while a_dot.dim() < tokens.dim():
        a_dot = a_dot.unsqueeze(-1)
        s_dot = s_dot.unsqueeze(-1)

    # velocity target
    v_target = a_dot * x0_recon + s_dot * noise

    return x_t, v_target


def train_step(
    model: Transformer3DModel,
    batch: dict,
    scheduler: RectifiedFlowScheduler,
    patchifier: SymmetricPatchifier,
    config: TrainConfig,
    device: torch.device = None,
    lpips_metric: any = None,
    decoder_vae: any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    model_dtype = next(model.parameters()).dtype

    latents = batch["latents"].to(device=device, dtype=model_dtype)
    face_embeds = batch["audio_latents"].to(
        device=device, dtype=model_dtype
    )  # [B, 256, D]
    audio_mask = batch.get("audio_mask", None)  # [B, 256]
    audio_mask = audio_mask.to(device)

    tokens, latent_coords = patchifier.patchify(latents)
    indices_grid = latent_coords

    B = tokens.shape[0]

    logn = torch.distributions.LogNormal(
        torch.tensor(config.rf_log_normal_mu, device=device),
        torch.tensor(config.rf_log_normal_sigma, device=device),
    )
    raw = logn.sample((B,))
    t_raw = raw / (1 + raw)
    t_low = torch.quantile(t_raw, config.rf_quantile_min)
    t_high = torch.quantile(t_raw, config.rf_quantile_max)
    t = t_raw.clamp(min=float(t_low), max=float(t_high))

    # Apply resolution-dependent shift
    samples_shape = tokens.view(B, -1, tokens.shape[-1]).shape
    t = scheduler.shift_timesteps(samples_shape, t)

    noise = torch.randn_like(tokens)
    noisy_tokens = scheduler.add_noise(
        original_samples=tokens, noise=noise, timesteps=t
    )
    noisy_tokens = noisy_tokens.to(dtype=model_dtype)

    x_t, v_target = build_velocity_target(tokens, noise, t, scheduler)
    v_target = v_target.to(dtype=model_dtype)

    out = model(
        hidden_states=noisy_tokens,
        indices_grid=indices_grid,
        encoder_hidden_states=face_embeds,
        timestep=t,
        attention_mask=None,
        encoder_attention_mask=audio_mask,
        return_dict=True,
    )
    std_target = v_target.std()
    transformer_mse = F.mse_loss(out.sample, v_target, reduction="mean")
    loss = float(getattr(config, "transformer_loss_weight", 1.0)) * transformer_mse

    # train decoder to perform the last denoising step in pixel space
    if getattr(config, "decoder_train", False) and (decoder_vae is not None):

        t_eps = 1e-6
        if t.ndim == 1:
            t_b = t.view(-1, 1, 1)
        else:
            t_b = t
        dt_small = torch.clamp(t_b, min=t_eps) * 0.05
        tokens_denoised = noisy_tokens - dt_small * out.sample

        t_max = float(getattr(config, "decoder_t_max", 0.1))
        t_d = torch.rand(B, device=device, dtype=t.dtype) * t_max

        # Unpatchify tokens to latent grids
        p_h, p_w = patchifier.patch_size[1], patchifier.patch_size[2]
        out_ch = tokens_denoised.shape[-1] // (p_h * p_w)
        latents_denoised = patchifier.unpatchify(
            latents=tokens_denoised,
            output_height=latents.shape[-2],
            output_width=latents.shape[-1],
            out_channels=out_ch,
        )

        denorm = _norm(
            latents_denoised, decoder_vae, vae_per_channel_normalize=True
        ).to(dtype=decoder_vae.dtype)
        dec_pixels = _vae_decode(
            denorm,
            decoder_vae,
            is_video=True,
            vae_per_channel_normalize=True,
            timestep=t_d,
        )

        Bv, Cpix, Fpix, Hpix, Wpix = dec_pixels.shape
        target_tensor = batch["target_video"].to(
            dec_pixels.device, dtype=dec_pixels.dtype
        )  # [C,F,H,W] per item
        # Ensure batch and shape alignment
        if target_tensor.ndim == 4:
            target_tensor = target_tensor.unsqueeze(0)
        target_tensor = target_tensor[:, :, :Fpix]
        target_tensor = torch.nn.functional.interpolate(
            target_tensor.flatten(0, 1),
            size=(Hpix, Wpix),
            mode="bilinear",
            align_corners=False,
        ).view(Bv, Cpix, -1, Hpix, Wpix)

        l1 = F.l1_loss(dec_pixels, target_tensor, reduction="mean")
        lpips_val = torch.tensor(0.0, device=dec_pixels.device, dtype=dec_pixels.dtype)
        if lpips_metric is not None:
            Bv, Cpix, Fpix, Hpix, Wpix = dec_pixels.shape
            acc = 0.0
            with torch.no_grad():
                for i in range(Fpix):
                    a = dec_pixels[:, :, i].clamp(0, 1).detach()
                    b = target_tensor[:, :, i].clamp(0, 1).detach()
                    acc = acc + lpips_metric(a, b)
            lpips_val = acc / max(1, Fpix)
            if torch.is_tensor(lpips_val) and lpips_val.ndim > 0:
                lpips_val = lpips_val.mean()
            lpips_val = torch.as_tensor(
                float(lpips_val), device=dec_pixels.device, dtype=dec_pixels.dtype
            )

        w_l1 = float(getattr(config, "decoder_loss_l1_weight", 0.1))
        w_lpips = float(getattr(config, "decoder_loss_lpips_weight", 0.0))
        dec_loss = w_l1 * l1 + w_lpips * lpips_val

        loss = loss + dec_loss
        loss_dict = {
            "transformer_mse": transformer_mse.detach().item(),
            "decoder_l1": l1.detach().item(),
            "decoder_lpips": (
                float(lpips_val.detach().item())
                if torch.is_tensor(lpips_val)
                else float(lpips_val)
            ),
            "decoder_loss": dec_loss.detach().item(),
        }

    else:
        loss_dict = {"transformer_mse": transformer_mse.detach().item()}
    rel_mse = loss / (std_target**2 + 1e-12)
    nrmse = torch.sqrt(loss) / (std_target + 1e-12)

    return loss, rel_mse, nrmse, loss_dict


def train_one_epoch(
    model: Transformer3DModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: RectifiedFlowScheduler,
    patchifier: SymmetricPatchifier,
    device: Union[torch.device, str],
    config: TrainConfig,
    epoch: int,
    global_step: int,
    decoder_vae: None,
):
    model.train()
    epoch_losses = []

    lpips_metric = LPIPS(net="vgg").to(device).eval()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        loss, rel_mse, nrmse, loss_dict = train_step(
            model,
            batch,
            scheduler,
            patchifier,
            config,
            device,
            lpips_metric,
            decoder_vae,
        )

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        loss_value = loss.item()

        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            actual_loss = loss_value * config.gradient_accumulation_steps
            log_payload = {
                "train/loss": actual_loss,
                "train/rel_mse": rel_mse.item(),
                "train/nrmse": nrmse.item(),
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            for k, v in (loss_dict or {}).items():
                log_payload[f"train/{k}"] = float(v)
            wandb.log(log_payload, step=global_step)

        epoch_losses.append(loss_value * config.gradient_accumulation_steps)

    epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return global_step, epoch_loss


def train_loop_deepspeed(config: TrainConfig, dataloader, val_dataloader=None):
    """
    Training loop with model sharding across GPUs.
    """

    patchifier = SymmetricPatchifier(patch_size=1)

    transformer = build_transformer(config)
    transformer = apply_training_strategy(
        transformer, config, getattr(config, "train_mode", "full")
    )

    if config.gradient_checkpointing:
        if hasattr(transformer, "base_model") and hasattr(
            transformer.base_model, "model"
        ):
            transformer.base_model.model.gradient_checkpointing = True
        else:
            try:
                transformer.gradient_checkpointing = True
            except Exception:
                pass
        print("Gradient checkpointing enabled")

    rf_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=config.rf_num_train_timesteps,
        shifting=config.rf_shifting,
        base_resolution=config.rf_base_resolution,
        target_shift_terminal=config.rf_target_shift_terminal,
        sampler=config.rf_sampler,
        shift=config.rf_shift,
    )
    opt_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optional decoder training: include decoder_vae params in optimizer
    decoder_vae = None
    if getattr(config, "decoder_train", False):
        base_ckpt = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=config.checkpoint_path,
            repo_type="model",
        )
        decoder_vae = CausalVideoAutoencoder.from_pretrained(base_ckpt).cpu()
        if config.precision == "bfloat16":
            decoder_vae = decoder_vae.to(torch.bfloat16)
        for p in decoder_vae.parameters():
            p.requires_grad = True
        opt_params += list(decoder_vae.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate)

    with open(config.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    combined_model = CombinedTrainModel(
        transformer=transformer, decoder_vae=decoder_vae
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=combined_model,
        optimizer=optimizer,
        config=ds_config,
    )

    device = model_engine.device

    # VAE from base checkpoint
    base_ckpt = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=config.checkpoint_path,
        repo_type="model",
    )
    vae = CausalVideoAutoencoder.from_pretrained(base_ckpt).cpu()
    val_components = {"vae": vae, "patchifier": patchifier, "scheduler": rf_scheduler}

    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)

    if model_engine.local_rank == 0:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "checkpoint_path": config.checkpoint_path,
                "precision": config.precision,
                "gradient_checkpointing": config.gradient_checkpointing,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "use_deepspeed": True,
                "world_size": model_engine.world_size,
            },
        )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(config.num_epochs or 0):

        model_engine.train()
        epoch_losses = []

        # Build LPIPS metric on the same device
        lpips_metric = LPIPS(net="vgg").to(device).eval()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch_device = {
                "latents": batch["latents"].to(device),
                "audio_latents": batch["audio_latents"].to(device),
                "audio_mask": (
                    batch.get("audio_mask").to(device)
                    if batch.get("audio_mask") is not None
                    else None
                ),
                "target_video": (
                    batch.get("target_video").to(device)
                    if batch.get("target_video") is not None
                    else None
                ),
            }

            loss, rel_mse, nrmse, loss_dict = train_step(
                (
                    model_engine.module
                    if hasattr(model_engine, "module")
                    else model_engine
                ),
                batch_device,
                rf_scheduler,
                patchifier,
                config,
                device,
                lpips_metric,
                (
                    model_engine.module.decoder_vae
                    if hasattr(model_engine, "module")
                    else None
                ),
            )

            model_engine.backward(loss)
            model_engine.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            global_step += 1

            if (
                model_engine.local_rank == 0
                and global_step % config.log_every_n_steps == 0
            ):
                log_payload = {
                    "train/loss": loss_value,
                    "train/rel_mse": float(rel_mse.item()),
                    "train/nrmse": float(nrmse.item()),
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for k, v in (loss_dict or {}).items():
                    log_payload[f"train/{k}"] = float(v)
                wandb.log(log_payload, step=global_step)
                print(
                    f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss_value:.6f}"
                )

        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        if model_engine.local_rank == 0 and val_dataloader is not None:
            val_loss = validate_epoch(
                model_engine, val_dataloader, rf_scheduler, patchifier, config, device
            )
            wandb.log({"val/loss": val_loss, "val/epoch": epoch}, step=global_step)
            print(f"Validation epoch {epoch+1}, loss: {val_loss:.6f}")

            val_components["vae"] = val_components["vae"].to(device)
            # Extract inner transformer for validation
            current_module = getattr(model_engine, "module", model_engine)
            current_transformer = getattr(current_module, "transformer", current_module)
            validate_video(
                transformer=current_transformer,
                components=val_components,
                val_dataloader=val_dataloader,
                output_dir=os.path.join(config.output_dir, "val_videos"),
                num_samples=min(2, len(val_dataloader.dataset)),
                frame_rate=22,
            )
            val_components["vae"] = val_components["vae"].cpu()

        if model_engine.local_rank == 0:
            print(f"Epoch {epoch+1} finished. Average loss: {epoch_loss:.6f}")
            wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)

        if (
            model_engine.local_rank == 0
            and (epoch + 1) % config.save_every_n_epochs == 0
        ):
            epoch_st = os.path.join(
                config.output_dir, f"model_epoch_{epoch+1}.safetensors"
            )
            # Unwrap and save transformer and decoder (if present)
            current_module = getattr(model_engine, "module", model_engine)
            current_transformer = getattr(current_module, "transformer", current_module)
            save_training_checkpoint(
                model=current_transformer,
                target_path=epoch_st,
                train_mode=getattr(config, "train_mode", "full"),
                metadata={
                    "epoch": str(epoch + 1),
                    "global_step": str(global_step),
                    "source": "deepspeed_epoch",
                    "scheduler": {
                        "num_train_timesteps": config.rf_num_train_timesteps,
                        "shifting": config.rf_shifting,
                        "base_resolution": config.rf_base_resolution,
                        "target_shift_terminal": config.rf_target_shift_terminal,
                        "sampler": config.rf_sampler,
                        "shift": config.rf_shift,
                    },
                    "vae": {"timestep_conditioning": True},
                },
                is_best=(epoch_loss < best_loss),
            )

            # Save decoder weights if training decoder is enabled
            if (
                getattr(config, "decoder_train", False)
                and hasattr(current_module, "decoder_vae")
                and (current_module.decoder_vae is not None)
            ):
                dec_path = os.path.join(
                    config.output_dir, f"decoder_epoch_{epoch+1}.safetensors"
                )
                save_module_safetensors(
                    current_module.decoder_vae,
                    dec_path,
                    metadata={
                        "epoch": str(epoch + 1),
                        "source": "deepspeed_epoch_decoder",
                    },
                )

    if model_engine.local_rank == 0:
        wandb.finish()
        print("Training complete!")


def train_loop(config: TrainConfig, dataloader, val_dataloader=None):
    if config.use_deepspeed:
        return train_loop_deepspeed(config, dataloader, val_dataloader)

    device = get_device()
    patchifier = SymmetricPatchifier(patch_size=1)

    model = build_transformer(config)
    model.to(device)
    model = apply_training_strategy(
        model, config, getattr(config, "train_mode", "full")
    )

    if config.gradient_checkpointing:
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model.base_model.model.gradient_checkpointing = True
        else:
            try:
                model.gradient_checkpointing = True
            except Exception:
                pass
        print("Gradient checkpointing enabled")

    rf_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=config.rf_num_train_timesteps,
        shifting=config.rf_shifting,
        base_resolution=config.rf_base_resolution,
        target_shift_terminal=config.rf_target_shift_terminal,
        sampler=config.rf_sampler,
        shift=config.rf_shift,
    )
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    decoder_vae = None
    if getattr(config, "decoder_train", False):
        base_ckpt = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=config.checkpoint_path,
            repo_type="model",
        )
        decoder_vae = CausalVideoAutoencoder.from_pretrained(base_ckpt).to(device)
        if config.precision == "bfloat16":
            decoder_vae = decoder_vae.to(torch.bfloat16)
        for p in decoder_vae.parameters():
            p.requires_grad = True
        params += list(decoder_vae.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

    total_params_transformer = sum(
        p.numel() for n, p in model.named_parameters() if "caption_projection" not in n
    )
    total_params_all = sum(p.numel() for _n, p in model.named_parameters())
    trainable_params = sum(
        p.numel() for _n, p in model.named_parameters() if p.requires_grad
    )
    trainable_params_transformer = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("caption_projection" not in n)
    )
    print(
        f"[params] total_all={total_params_all} total_transformer={total_params_transformer} trainable_all={trainable_params} trainable_transformer={trainable_params_transformer}"
    )

    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "checkpoint_path": config.checkpoint_path,
            "precision": config.precision,
            "rf_num_train_timesteps": config.rf_num_train_timesteps,
            "rf_sampler": config.rf_sampler,
        },
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    wandb.run.summary["trainable_params"] = trainable_params
    wandb.run.summary["total_params"] = total_params

    global_step = 0
    best_loss = float("inf")

    # Preload shared components for validation
    base_ckpt = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=config.checkpoint_path,
        repo_type="model",
    )
    vae = CausalVideoAutoencoder.from_pretrained(base_ckpt).cpu()
    val_components = {"vae": vae, "patchifier": patchifier, "scheduler": rf_scheduler}

    for epoch in range(config.num_epochs or 0):
        global_step, epoch_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            rf_scheduler,
            patchifier,
            device,
            config,
            epoch,
            global_step,
            decoder_vae,
        )

        if val_dataloader is not None:
            val_loss = validate_epoch(
                model, val_dataloader, rf_scheduler, patchifier, config, device
            )
            wandb.log({"val/loss": val_loss, "val/epoch": epoch}, step=global_step)
            print(f"Validation epoch {epoch+1}, loss: {val_loss:.6f}")

            val_components["vae"] = val_components["vae"].to(device)
            current_transformer = getattr(
                getattr(model, "base_model", model), "model", model
            )
            validate_video(
                transformer=current_transformer,
                components=val_components,
                val_dataloader=val_dataloader,
                output_dir=os.path.join(config.output_dir, "val_videos"),
                num_samples=1,
                frame_rate=22,
            )
            val_components["vae"] = val_components["vae"].cpu()
        print(f"Epoch {epoch+1} finished. Average loss: {epoch_loss:.6f}")
        wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)

        if (epoch + 1) % config.save_every_n_epochs == 0:
            epoch_st = os.path.join(
                config.output_dir, f"model_epoch_{epoch+1}.safetensors"
            )
            save_training_checkpoint(
                model=model,
                target_path=epoch_st,
                train_mode=getattr(config, "train_mode", "full"),
                metadata={
                    "epoch": str(epoch + 1),
                    "global_step": str(global_step),
                    "source": "single_gpu_epoch",
                    "scheduler": {
                        "num_train_timesteps": config.rf_num_train_timesteps,
                        "shifting": config.rf_shifting,
                        "base_resolution": config.rf_base_resolution,
                        "target_shift_terminal": config.rf_target_shift_terminal,
                        "sampler": config.rf_sampler,
                        "shift": config.rf_shift,
                    },
                    "vae": {"timestep_conditioning": True},
                },
                is_best=epoch_loss < best_loss,
            )

            # Save decoder weights if training decoder is enabled
            if getattr(config, "decoder_train", False) and (decoder_vae is not None):
                dec_path = os.path.join(
                    config.output_dir, f"decoder_epoch_{epoch+1}.safetensors"
                )
                save_module_safetensors(
                    decoder_vae,
                    dec_path,
                    metadata={
                        "epoch": str(epoch + 1),
                        "source": "single_gpu_epoch_decoder",
                    },
                )

    wandb.finish()
    print("Training complete!")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():

    parser = argparse.ArgumentParser(
        description="LTX-Video transformer training entry point"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config with train block (e.g., configs/train-avatars.yaml)",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["full", "lora_audio"],
        default="full",
        help="Training mode: full finetune or LoRA+AudioProjection only",
    )
    # Accept and ignore any extra args injected by launchers (DeepSpeed, torchrun, etc.)
    args, _unknown = parser.parse_known_args()

    config = load_train_config_from_yaml(args.config)
    setattr(config, "train_mode", args.train_mode)

    dataset = LatentPairDataset(
        config.audio_latents_dir, config.encoder_latents_dir, config.videos
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size or 1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_latent_pairs,
    )
    val_dataset = ValidationDataset(
        config.val_audio_latents_dir, config.val_encoder_latents_dir, config.videos
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size or 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_latent_pairs,
    )

    train_loop(config, dataloader, val_dataloader)


if __name__ == "__main__":
    main()
