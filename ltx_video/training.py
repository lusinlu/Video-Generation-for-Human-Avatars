import os
import argparse
from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel, T5Tokenizer
from ltx_video.utils.torch_utils import (
    save_training_checkpoint,
)
from ltx_video.config import TrainConfig, load_train_config_from_yaml
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import (
    Transformer3DModel,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.dataset import LatentPairDataset, ValidationDataset, collate_latent_pairs
from ltx_video.validation import validate_epoch


def build_transformer(config: TrainConfig, patchifier: SymmetricPatchifier = None):
    ltxv_model_path = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=config.checkpoint_path,
        repo_type="model",
    )
    if config.precision == "bfloat16":
        model = Transformer3DModel.from_pretrained(
            ltxv_model_path, patchifier=patchifier
        ).to(torch.bfloat16)
    else:
        model = Transformer3DModel.from_pretrained(
            ltxv_model_path, patchifier=patchifier
        )

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
        # for _n, p in model.named_parameters():
        #     p.requires_grad = True
        for n, p in model.named_parameters():
            trainable = any(
                key in n
                for key in (
                    "proj_out",
                    "scale_shift_table",
                    "adaln_single",
                    "caption_projection",
                    "attn",  # self-attn layers
                    "attn2",  # cross-attn layers
                )
            )
            p.requires_grad = trainable
        return model


def train_step(
    model: Transformer3DModel,
    batch: dict,
    scheduler: RectifiedFlowScheduler,
    patchifier: SymmetricPatchifier,
    config: TrainConfig,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    model_dtype = next(model.parameters()).dtype

    latents = batch["latents"].to(device=device, dtype=model_dtype)
    ref_image_latents = batch["ref_image_latents"].to(device=device, dtype=model_dtype)
    pose_latents = batch["pose_latents"].to(device=device, dtype=model_dtype)

    # Use encoded prompt as encoder_hidden_states (same as inference)
    # Expand prompt_embeds to match batch size
    batch_size = latents.shape[0]
    encoder_hidden_states = prompt_embeds.expand(batch_size, -1, -1).to(
        device=device, dtype=model_dtype
    )  # [B, 256, D]

    encoder_attention_mask = prompt_attention_mask.expand(batch_size, -1).to(device)

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

    v_target = scheduler.build_velocity_target(tokens, noise, t)
    v_target = v_target.to(dtype=model_dtype)

    out = model(
        hidden_states=noisy_tokens,
        indices_grid=indices_grid,
        ref_image_hidden_states=ref_image_latents,
        pose_hidden_states=pose_latents,
        encoder_hidden_states=encoder_hidden_states,
        timestep=t,
        attention_mask=None,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    std_target = v_target.std()
    transformer_mse = F.mse_loss(out.sample, v_target, reduction="mean")
    loss = float(getattr(config, "transformer_loss_weight", 1.0)) * transformer_mse
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
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    epoch: int,
    global_step: int,
):
    model.train()
    epoch_losses = []

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        loss, rel_mse, nrmse, loss_dict = train_step(
            model,
            batch,
            scheduler,
            patchifier,
            config,
            prompt_embeds,
            prompt_attention_mask,
            device,
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


def train_loop(
    config: TrainConfig,
    dataloader,
    val_dataloader=None,
    prompt_embeds: torch.Tensor = None,
    prompt_attention_mask: torch.Tensor = None,
):
    if config.use_deepspeed:
        from ltx_video.training_deepspeed import train_loop_deepspeed

        return train_loop_deepspeed(config, dataloader, val_dataloader)

    device = get_device()
    patchifier = SymmetricPatchifier(patch_size=1)
    model = build_transformer(config, patchifier=patchifier)
    model.to(device)
    model = apply_training_strategy(
        model, config, getattr(config, "train_mode", "full")
    )

    # if val_dataloader is not None:
    #     lpips_metric = LPIPS(net="vgg").to(device).eval()
    #     fid_metric = FrechetInceptionDistance(normalize=True).to(device)

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
    # base_ckpt = hf_hub_download(
    #     repo_id="Lightricks/LTX-Video",
    #     filename=config.checkpoint_path,
    #     repo_type="model",
    # )
    # vae = CausalVideoAutoencoder.from_pretrained(base_ckpt).cpu()
    # val_components = {"vae": vae, "patchifier": patchifier, "scheduler": rf_scheduler}

    for epoch in range(config.num_epochs or 0):
        global_step, epoch_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            rf_scheduler,
            patchifier,
            device,
            config,
            prompt_embeds,
            prompt_attention_mask,
            epoch,
            global_step,
        )

        if val_dataloader is not None:
            val_loss = validate_epoch(
                model,
                val_dataloader,
                rf_scheduler,
                patchifier,
                config,
                prompt_embeds,
                prompt_attention_mask,
                device,
            )
            wandb.log({"val/loss": val_loss, "val/epoch": epoch}, step=global_step)
            print(f"Validation epoch {epoch+1}, loss: {val_loss:.6f}")

            # val_components["vae"] = val_components["vae"].to(device)
            # current_transformer = getattr(
            #     getattr(model, "base_model", model), "model", model
            # )
            # validate_video(
            #     transformer=current_transformer,
            #     components=val_components,
            #     val_dataloader=val_dataloader,
            #     output_dir=os.path.join(config.output_dir, "val_videos"),
            #     num_samples=1,
            #     frame_rate=22,
            #     lpips_metric=lpips_metric,
            #     fid_metric=fid_metric,
            # )
            # val_components["vae"] = val_components["vae"].cpu()
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

    wandb.finish()
    print("Training complete!")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_prompt(
    prompt: str,
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    device: torch.device,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a prompt using T5 tokenizer and encoder, same as pipeline.encode_prompt.

    Args:
        prompt: Text prompt to encode
        tokenizer: T5Tokenizer instance
        text_encoder: T5EncoderModel instance
        device: Device to place tensors on
        max_length: Maximum sequence length (default 256)

    Returns:
        Tuple of (prompt_embeds, prompt_attention_mask)
        - prompt_embeds: [1, max_length, hidden_dim]
        - prompt_attention_mask: [1, max_length]
    """
    text_enc_device = next(text_encoder.parameters()).device

    # Tokenize the prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask

    # Move to text encoder device
    prompt_attention_mask = prompt_attention_mask.to(text_enc_device)

    # Encode with text encoder
    prompt_embeds = text_encoder(
        text_input_ids.to(text_enc_device),
        attention_mask=prompt_attention_mask,
    )
    prompt_embeds = prompt_embeds[0]  # Get hidden states

    # Move to target device and dtype
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.to(device)

    return prompt_embeds, prompt_attention_mask


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
        config.condition_latents_dir,
        config.encoder_latents_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_latent_pairs,
    )
    val_dataset = ValidationDataset(
        config.val_condition_latents_dir, config.val_encoder_latents_dir
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_latent_pairs,
    )

    prompt = "Person speaking naturally, with natual face and body movements"

    # Load text encoder and tokenizer (same as inference.py)
    # Default text encoder path (can be overridden in config)
    text_encoder_model_name_or_path = getattr(
        config, "text_encoder_model_name_or_path", "PixArt-alpha/PixArt-XL-2-1024-MS"
    )

    device = get_device()
    print(f"Loading text encoder from {text_encoder_model_name_or_path}...")
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_model_name_or_path, subfolder="text_encoder"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder = text_encoder.to(device)
    text_encoder = text_encoder.to(torch.bfloat16)
    text_encoder.eval()

    # Encode the prompt
    print(f"Encoding prompt: {prompt}")
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = encode_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )

    print(f"Prompt encoded. Shape: {prompt_embeds.shape}")

    train_loop(config, dataloader, val_dataloader, prompt_embeds, prompt_attention_mask)


if __name__ == "__main__":
    main()
