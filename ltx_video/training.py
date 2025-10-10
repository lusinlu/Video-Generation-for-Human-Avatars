import os
import argparse
import json
from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import json
from ltx_video.config import TrainConfig, load_train_config_from_yaml
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel, AudioProjection
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.dataset import LatentPairDataset, collate_latent_pairs
try:
    import deepspeed
    from deepspeed.utils import logger as ds_logger
except ImportError:
    raise ImportError(
        "DeepSpeed not installed! Install with: pip install deepspeed\n"
        "Or set use_deepspeed: false in your config."
    )

def build_transformer_for_lora_finetune(config: TrainConfig):
    ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=config.checkpoint_path,
            repo_type="model",
        )
    if config.precision == "bfloat16":
        model = Transformer3DModel.from_pretrained(ltxv_model_path).to(torch.bfloat16)
    else:
        model = Transformer3DModel.from_pretrained(ltxv_model_path)
    
    model.caption_projection = AudioProjection(
        in_features=config.audio_embed_dim,
        hidden_size=model.inner_dim
    )
    
    if config.precision == "bfloat16":
        model.caption_projection = model.caption_projection.to(torch.bfloat16)
        
    return model


def apply_lora_to_model(model: torch.nn.Module, config: TrainConfig):
    """
    Apply LoRA to cross-attention layers using PEFT library.
    """
    # Make caption_projection modul trainable
    if hasattr(model, "caption_projection") and model.caption_projection is not None:
        for p in model.caption_projection.parameters():
            p.requires_grad = True
    
    # Configure LoRA to target only cross-attention layers (attn2)
    target_modules = []
    for i in range(len(model.transformer_blocks)):
        target_modules.extend([
            f"transformer_blocks.{i}.attn2.to_q",
            f"transformer_blocks.{i}.attn2.to_k", 
            f"transformer_blocks.{i}.attn2.to_v",
            f"transformer_blocks.{i}.attn2.to_out.0",
        ])
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    return model


def compute_alpha_sigma(scheduler, t: torch.Tensor):
    sigma = t
    alpha = 1 - sigma
    return alpha, sigma


def build_velocity_target(tokens: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, scheduler, dt_eps: float = 1e-4):
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
) -> torch.Tensor:
    model_dtype = next(model.parameters()).dtype
    
    latents = batch["latents"].to(device=device, dtype=model_dtype)
    face_embeds = batch["audio_latents"].to(device=device, dtype=model_dtype)  # [B, 256, D]
    audio_mask = batch.get("audio_mask", None)  # [B, 256]
    if audio_mask is not None:
        audio_mask = audio_mask.to(device)
    
    tokens, latent_coords = patchifier.patchify(latents)
    indices_grid = latent_coords
    

    B = tokens.shape[0]

    if config.rf_log_normal_mu is not None and config.rf_log_normal_sigma is not None:
        logn = torch.distributions.LogNormal(
            torch.tensor(config.rf_log_normal_mu, device=device),
            torch.tensor(config.rf_log_normal_sigma, device=device)
        )
        raw = logn.sample((B,))
        t_raw = raw / (1 + raw)
        qmin = config.rf_quantile_min
        qmax = config.rf_quantile_max
        t_low = torch.quantile(t_raw, qmin)
        t_high = torch.quantile(t_raw, qmax)
        t = t_raw.clamp(min=float(t_low), max=float(t_high))
    else:
        t = torch.rand((B,), device=device).clamp(min=1e-6, max=1.0)

    # Apply resolution-dependent shift
    samples_shape = tokens.view(B, -1, tokens.shape[-1]).shape
    t = scheduler.shift_timesteps(samples_shape, t)

    noise = torch.randn_like(tokens)
    noisy_tokens = scheduler.add_noise(original_samples=tokens, noise=noise, timesteps=t)
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
    pred = out.sample

    if hasattr(scheduler, "weight"):
        w = scheduler.weight(t)
        while w.dim() < pred.dim():
            w = w.unsqueeze(-1)
        mse_per_sample = ((pred - v_target) ** 2).mean(dim=list(range(1, pred.dim())))
        loss = (w * mse_per_sample).mean()
    else:
        loss = F.mse_loss(pred, v_target, reduction="mean")

    return loss



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
):
    model.train()
    epoch_losses = []
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, batch, scheduler, patchifier, config, device)
        
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        
        loss_value = loss.item()
        
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            
            actual_loss = loss_value * config.gradient_accumulation_steps 
            wandb.log({
                "train/loss": actual_loss,
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)
            
        
        epoch_losses.append(loss_value * config.gradient_accumulation_steps)
    
    epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return global_step, epoch_loss


def train_loop_deepspeed(config: TrainConfig, dataloader):
    """
    Training loop with model sharding across GPUs.
    """
    
    patchifier = SymmetricPatchifier(patch_size=1)
    
    model = build_transformer_for_lora_finetune(config)
    model = apply_lora_to_model(model, config)
    
    if config.gradient_checkpointing:
        model.base_model.model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")
    
    rf_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=config.rf_num_train_timesteps,
        shifting=config.rf_shifting,
        base_resolution=config.rf_base_resolution,
        target_shift_terminal=config.rf_target_shift_terminal,
        sampler=config.rf_sampler,
        shift=config.rf_shift,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )
    
    with open(config.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
   
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    
    device = model_engine.local_rank
    patchifier.to(device)
    
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
    
    if model_engine.local_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        wandb.run.summary["trainable_params"] = trainable_params
        wandb.run.summary["total_params"] = total_params
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs or 0):
        
        model_engine.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            latents = batch["latents"].to(device)
            face_embeds = batch["audio_latents"].to(device)
            audio_mask = batch.get("audio_mask")
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            
            batch_device = {"latents": latents, "audio_latents": face_embeds, "audio_mask": audio_mask}
            
            loss = train_step(model_engine, batch_device, rf_scheduler, patchifier, config, device)
            
            model_engine.backward(loss)
            model_engine.step()
            
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            global_step += 1
            
            if model_engine.local_rank == 0 and global_step % config.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": loss_value,
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)
                print(
                    f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss_value:.6f}"
                )
        
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        
        if model_engine.local_rank == 0:
            print(f"Epoch {epoch+1} finished. Average loss: {epoch_loss:.6f}")
            wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)
        
        if model_engine.local_rank == 0 and (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = os.path.join(config.output_dir, f"ckpt_epoch_{epoch+1}.pt")
            merged_model = model.merge_and_unload()
            torch.save({
                "model": merged_model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
        
        # Save best model (only rank 0)
        if model_engine.local_rank == 0 and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(config.output_dir, "ckpt_best.pt")
            merged_model = model.merge_and_unload()
            torch.save({
                "model": merged_model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }, best_path)
            logger.info(f"Saved best checkpoint (loss: {best_loss:.6f}): {best_path}")
    
    if model_engine.local_rank == 0:
        wandb.finish()
        logger.info("Training complete!")


def train_loop(config: TrainConfig, dataloader):
    if config.use_deepspeed:
        return train_loop_deepspeed(config, dataloader)
    
    device = get_device()
    patchifier = SymmetricPatchifier(patch_size=1)
    
    model = build_transformer_for_lora_finetune(config)
    model.to(device)
    model = apply_lora_to_model(model, config)
    
    if config.gradient_checkpointing:
        model.base_model.model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")

    rf_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=config.rf_num_train_timesteps,
        shifting=config.rf_shifting,
        base_resolution=config.rf_base_resolution,
        target_shift_terminal=config.rf_target_shift_terminal,
        sampler=config.rf_sampler,
        shift=config.rf_shift,
    )
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)

   
    wandb.init(project=config.wandb_project,
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
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]

    wandb.run.summary["trainable_params"] = trainable_params
    wandb.run.summary["total_params"] = total_params

    global_step = 0
    best_loss = float('inf')

    for epoch in range(config.num_epochs or 0):        
        global_step, epoch_loss = train_one_epoch(
            model, dataloader, optimizer, rf_scheduler, patchifier, device, config, epoch, global_step
        )
        
        print(f"Epoch {epoch+1} finished. Average loss: {epoch_loss:.6f}")
        wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)

        
        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = os.path.join(config.output_dir, f"ckpt_epoch_{epoch+1}.pt")
            
            model.save_pretrained(os.path.join(config.output_dir, f"lora_epoch_{epoch+1}"))
            torch.save({
                "lora_state_dict": get_peft_model_state_dict(model),
                "caption_projection": model.base_model.model.caption_projection.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(config.output_dir, "ckpt_best.pt")
            
            model.save_pretrained(os.path.join(config.output_dir, "lora_best"))
            torch.save({
                "lora_state_dict": get_peft_model_state_dict(model),
                "caption_projection": model.base_model.model.caption_projection.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }, best_path)
            print(f"Best checkpoint saved: {best_path}")
          
    wandb.finish()
    print("Training complete!")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    
    parser = argparse.ArgumentParser(description="LTX-Video transformer LoRA training entry point")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config with train block (e.g., configs/train-avatars.yaml)")
    args = parser.parse_args()

    config = load_train_config_from_yaml(args.config)
    
    dataset = LatentPairDataset(config.audio_latents_dir, config.encoder_latents_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size or 1, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_latent_pairs, 
    )
    
    train_loop(config, dataloader)


if __name__ == "__main__":
    main()




