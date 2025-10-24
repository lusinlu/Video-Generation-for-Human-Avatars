import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import hf_hub_download

import imageio
from PIL import Image

import torch
from torch import Tensor
import torchvision.transforms.functional as TVF

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode


def read_video(path: str) -> Tuple[List[Image.Image], float]:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 25.0))
    except Exception:
        fps = 25.0
    frames: List[Image.Image] = []
    for fr in reader:
        frames.append(Image.fromarray(fr).convert("RGB"))
    reader.close()
    return frames, fps


def preprocess_frames(frames: List[Image.Image], height: int, width: int) -> Tensor:
    processed = []
    for im in frames:
        im2 = im.resize((width, height), Image.BICUBIC)
        t = TVF.to_tensor(im2)  # [0,1]
        t = (t * 2.0) - 1.0     # [-1,1]
        processed.append(t)
    if not processed:
        raise ValueError("No frames to process")
    # Stack to (F, C, H, W) → (1, C, F, H, W)
    x = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return x


def iter_clips(num_frames: int, clip_length: int, stride: int) -> List[Tuple[int, int]]:
    clips: List[Tuple[int, int]] = []
    if num_frames <= 0:
        return clips
    i = 0
    while i < num_frames:
        j = i + clip_length
        if j > num_frames:
            break
        clips.append((i, j))  # [i, j)
        if j == num_frames:
            break
        i += max(1, stride)
    return clips


def save_latents_and_meta(
    latents: Tensor,
    out_dir: str,
    base_name: str,
    clip_idx: int,
    start_f: int,
    end_f: int,
    fps: float,
    vae_per_channel_normalize: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    lat_path = os.path.join(out_dir, f"{base_name}_{clip_idx}.pt")
    torch.save({"latents": latents.cpu()}, lat_path)

    meta = {
        "video": base_name,
        "clip_index": clip_idx,
        "start_frame": int(start_f),
        "end_frame_exclusive": int(end_f),
        "fps": float(fps),
        "start_time_sec": float(start_f / max(fps, 1e-8)),
        "end_time_sec": float(end_f / max(fps, 1e-8)),
        "vae_per_channel_normalize": bool(vae_per_channel_normalize),
        "format": "torch.pt",
    }
    with open(os.path.join(out_dir, f"{base_name}_{clip_idx}.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_vae(ckpt_path: str, device: torch.device) -> CausalVideoAutoencoder:
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    vae = vae.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    vae.eval().to(device)
    return vae


def main():
    parser = argparse.ArgumentParser(description="Extract normalized VAE latents from videos with clip metadata.")
    parser.add_argument("--inputs", type=str, nargs="+", help="Video files or a directory (processed recursively if dir)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default='ltxv-2b-0.9.6-dev-04-25.safetensors', help="Path to LTX-Video checkpoint (safetensors or dir)")
    parser.add_argument("--clip_length", type=int, default=121)
    parser.add_argument("--stride", type=int, default=121, help="Frames to move between clips")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--per_channel_normalize", action="store_true", help="Use per-channel VAE normalization")
    args = parser.parse_args()

    resolved_ckpt = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=str(args.ckpt),
        repo_type="model",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    vae = load_vae(resolved_ckpt, device=device)

    files: List[str] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi"):
                files.extend([str(pp) for pp in p.rglob(ext)])
        else:
            files.append(str(p))

    for vid_path in files:
        frames, fps = read_video(vid_path)
        if not frames:
            continue
        base = os.path.splitext(os.path.basename(vid_path))[0]
        clips = iter_clips(len(frames), args.clip_length, args.stride)
        if not clips:
            continue

        clip_idx = 0
        for (s, e) in clips:
            sub = frames[s:e]
            x = preprocess_frames(sub, args.height, args.width)
            x = x.to(device=vae.device, dtype=vae.dtype)
            with torch.no_grad():
                lat = vae_encode(x, vae, split_size=1, vae_per_channel_normalize=args.per_channel_normalize)
            save_latents_and_meta(
                latents=lat,
                out_dir=args.output_dir,
                base_name=base,
                clip_idx=clip_idx,
                start_f=s,
                end_f=e,
                fps=fps,
                vae_per_channel_normalize=args.per_channel_normalize,
            )
            clip_idx += 1


if __name__ == "__main__":
    main()


