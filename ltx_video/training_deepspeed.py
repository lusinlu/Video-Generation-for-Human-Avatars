import os
import json
from lpips import LPIPS
import torch
from huggingface_hub import hf_hub_download
from ltx_video.utils.torch_utils import (
    save_training_checkpoint,
)
from ltx_video.config import TrainConfig
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.validation import validate_epoch, validate_video
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.training import (
    build_transformer,
    apply_training_strategy,
    train_step,
)

try:
    import deepspeed
except ImportError:
    raise ImportError(
        "DeepSpeed not installed! Install with: pip install deepspeed\n"
        "Or set use_deepspeed: false in your config."
    )


class CombinedTrainModel(torch.nn.Module):
    """
    Wraps the transformer into a module for DeepSpeed.
    The forward proxies to the transformer.
    """

    def __init__(
        self,
        transformer: Transformer3DModel,
    ):
        super().__init__()
        self.transformer = transformer

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


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

    optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate)

    with open(config.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    combined_model = CombinedTrainModel(transformer=transformer)

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
        import wandb
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
                    model_engine.module.transformer
                    if hasattr(model_engine, "module")
                    else model_engine.transformer
                ),
                batch_device,
                rf_scheduler,
                patchifier,
                config,
                device,
                lpips_metric,
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
                import wandb
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
            import wandb
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
            import wandb
            wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)

        if (
            model_engine.local_rank == 0
            and (epoch + 1) % config.save_every_n_epochs == 0
        ):
            epoch_st = os.path.join(
                config.output_dir, f"model_epoch_{epoch+1}.safetensors"
            )
            # Unwrap and save transformer
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

    if model_engine.local_rank == 0:
        import wandb
        wandb.finish()
        print("Training complete!")

