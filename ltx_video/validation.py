import torch
import torch.nn.functional as F
from typing import Union

from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from safetensors import safe_open
import json
import imageio
import numpy as np
from pathlib import Path
from lpips import LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader,
    scheduler: RectifiedFlowScheduler,
    patchifier: SymmetricPatchifier,
    config,
    device: Union[torch.device, str],
):
    """Run one validation pass and return average MSE on velocity target.

    Mirrors train_step distributions for t (lognormal with quantile clamp),
    computes velocity targets and evaluates model prediction loss.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    for step_idx, batch in enumerate(dataloader):
        # Move batch
        latents = batch["latents"].to(device)
        face_embeds = batch["audio_latents"].to(device)
        audio_mask = batch.get("audio_mask")
        if audio_mask is not None:
            audio_mask = audio_mask.to(device)

        # Patchify latents
        tokens, latent_coords = patchifier.patchify(latents)
        indices_grid = latent_coords

        B = tokens.shape[0]
        mu, sigma = float(config.rf_log_normal_mu), float(config.rf_log_normal_sigma)
        logn = torch.distributions.LogNormal(
            torch.tensor(config.rf_log_normal_mu, device=device),
            torch.tensor(config.rf_log_normal_sigma, device=device),
        )
        raw = logn.sample((B,))
        t_raw = raw / (1 + raw)
        t_low = torch.quantile(t_raw, config.rf_quantile_min)
        t_high = torch.quantile(t_raw, config.rf_quantile_max)
        t = t_raw.clamp(min=float(t_low), max=float(t_high))

        # Resolution-dependent shift
        samples_shape = tokens.view(B, -1, tokens.shape[-1]).shape
        t = scheduler.shift_timesteps(samples_shape, t)

        # Build velocity target
        noise = torch.randn_like(tokens)
        # x_t and v_target (targets in same dtype as model
        alpha = scheduler.alpha(t)
        sigma = scheduler.sigma(t)
        while alpha.dim() < tokens.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        x_t = alpha * tokens + sigma * noise
        x0_recon = (x_t - sigma * noise) / (alpha + 1e-12)
        a_dot = scheduler.alpha_dot(t)
        s_dot = scheduler.sigma_dot(t)
        while a_dot.dim() < tokens.dim():
            a_dot = a_dot.unsqueeze(-1)
            s_dot = s_dot.unsqueeze(-1)
        v_target = a_dot * x0_recon + s_dot * noise

        model_dtype = next(model.parameters()).dtype
        x_t = x_t.to(dtype=model_dtype)
        v_target = v_target.to(dtype=model_dtype)

        # Forward
        out = model(
            hidden_states=x_t,
            indices_grid=indices_grid,
            encoder_hidden_states=face_embeds.to(dtype=model_dtype),
            timestep=t,
            attention_mask=None,
            encoder_attention_mask=audio_mask,
            return_dict=True,
        )

        loss = F.mse_loss(out.sample, v_target, reduction="mean")
        total_loss += float(loss.item())
        total_count += 1

    return total_loss / max(1, total_count)
    
def build_val_pipeline(transformer: torch.nn.Module, components: dict, device: torch.device):
    """Most efficient approach: reuse preloaded VAE/patchifier/scheduler passed in components,
    and rebuild a minimal pipeline each epoch with the current transformer instance.
    components must include: 'vae' (CausalVideoAutoencoder), 'patchifier' (SymmetricPatchifier), 'scheduler' (RectifiedFlowScheduler)
    """
    vae = components["vae"].to(device)
    patchifier = components["patchifier"]
    scheduler = components["scheduler"]
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "scheduler": scheduler,
        "vae": vae,
        "allowed_inference_steps": None,
    }
    pl = LTXVideoPipeline(**submodel_dict)
    pl = pl.to(device)
    return pl



@torch.no_grad()
def validate_video(
    transformer: torch.nn.Module,
    components: dict,
    val_dataloader,
    output_dir: str,
    num_samples: int = 4,
    frame_rate: int = 22,
):
    """Sample N items from val_dataloader, run full reconstruction with the provided pipeline (same path as inference),
    save videos, and compute LPIPS + FID vs target videos.

    - Assumes each batch item includes keys: 'latents', 'audio_latents', 'target_video_path', 'stem'.
    - Pipeline is called similarly to inference: we pass height/width/num_frames derived from latents.
    """
    print("[validate_video] starting")
    device = next(transformer.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lpips_metric = LPIPS(net='vgg').to(device)
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    taken = 0
    for batch_idx, batch in enumerate(val_dataloader):
        print(f"[validate_video] batch={batch_idx}")
        if taken >= num_samples:
            break
        # Use the first item in batch for deterministic visualization
        latents = batch["latents"][0:1].to(device)
        audio_latents = batch["audio_latents"][0:1].to(device)
        audio_mask = batch.get("audio_mask", None)
        if audio_mask is not None:
            audio_mask = audio_mask[0:1].to(device)
        stem = batch["stem"][0]
        target_path = batch["target_video_path"][0]
        


        # Derive dims from latents
        B, C, F, H, W = latents.shape
        # Build a fresh lightweight pipeline for current transformer
        pl = build_val_pipeline(transformer, components, device)
        height = H * pl.vae_scale_factor
        width = W * pl.vae_scale_factor
   
        video_scale_factor = getattr(pl, 'video_scale_factor', 1)
        num_frames = (F - 1) * video_scale_factor
        print(f"[validate_video] dims: H={height} W={width} F={num_frames} (latent_frames={F})")

        original_encode_prompt = pl.encode_prompt
        
        def patched_encode_prompt(prompt, **kwargs):
            return audio_latents.to(pl.device, dtype=pl.transformer.dtype), audio_mask.to(pl.device) if audio_mask is not None else torch.ones((audio_latents.shape[0], audio_latents.shape[1]), dtype=torch.long, device=pl.device)
        
        pl.encode_prompt = patched_encode_prompt
        
        try:
            result = pl(
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                prompt="dummy",  
                num_inference_steps=40,
                latents=latents.to(pl.device, dtype=pl.transformer.dtype),
                media_items=None,
                output_type="pt",
                is_video=True,
                vae_per_channel_normalize=True,
                mixed_precision=(pl.transformer.dtype == torch.bfloat16),
                offload_to_cpu=False,
                guidance_scale=1.0,
                stg_scale=0.0,
                rescaling_scale=1.0,
            )
        finally:
            # Restore original method
            pl.encode_prompt = original_encode_prompt
        recon = result.images  # [B, C, F, H, W]
        recon = recon[0].permute(1, 2, 3, 0).cpu().float().numpy()  # F,H,W,C uint8 below
        recon = (recon * 255).astype(np.uint8)

        # Save recon
        out_path = Path(output_dir) / f"val_recon_{stem}.mp4"
        with imageio.get_writer(str(out_path), fps=frame_rate) as writer:
            for fr in recon:
                writer.append_data(fr)

        # Load target frames
        reader = imageio.get_reader(target_path)
        tgt_frames = []
        for i in range(min(len(recon), reader.count_frames())):
            tgt_frames.append(reader.get_data(i))
        reader.close()
        if len(tgt_frames) == 0:
            print("[validate_video][warn] target has no frames; skipping metrics")
            continue
        tgt = np.stack(tgt_frames, axis=0)  # F,H,W,C

        # Resize to match recon if needed
        if tgt.shape[1:3] != recon.shape[1:3]:
            # simple center crop or resize can be added; here we take min dims and center-crop
            h = min(tgt.shape[1], recon.shape[1])
            w = min(tgt.shape[2], recon.shape[2])
            tgt = tgt[:, :h, :w]
            recon = recon[:, :h, :w]

        # LPIPS: compute per-frame and average
        lpips_vals = []
        for i in range(min(tgt.shape[0], recon.shape[0])):
            a = torch.from_numpy(recon[i]).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            b = torch.from_numpy(tgt[i]).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            lpips_val = float(lpips_metric(a*2-1, b*2-1).item())
            lpips_vals.append(lpips_val)
        avg_lpips = sum(lpips_vals) / max(1, len(lpips_vals))

        # FID: add all frames to metric
        # Convert to N,C,H,W in [0,1]
        recon_tensor = torch.from_numpy(recon).permute(0, 3, 1, 2).to(device).float() / 255.0
        tgt_tensor = torch.from_numpy(tgt).permute(0, 3, 1, 2).to(device).float() / 255.0
        fid_metric.update(recon_tensor, real=False)
        fid_metric.update(tgt_tensor, real=True)

        fid_val = float(fid_metric.compute().item())

        print(f"[val video] {stem}: lpips={avg_lpips:.4f}, fid={fid_val:.4f}, saved={out_path}")
        taken += 1

