import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from torch import Tensor
import torchvision.transforms.functional as TVF
from huggingface_hub import hf_hub_download
from PIL import Image

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import vae_encode


def preprocess_frames(frames: List[Image.Image], height: int, width: int) -> Tensor:
    """Preprocess frames same as save_vae_latents.py."""
    processed = []
    for im in frames:
        im2 = im.resize((width, height), Image.BICUBIC)
        t = TVF.to_tensor(im2)  # [0,1]
        t = (t * 2.0) - 1.0  # [-1,1]
        processed.append(t)
    if not processed:
        raise ValueError("No frames to process")
    # Stack to (F, C, H, W) → (1, C, F, H, W)
    x = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return x


def load_pose_frames(poses_dir: Path, target_length: int = 57) -> List[Image.Image]:
    """
    Load pose frames from directory and pad/truncate to target_length.

    Args:
        poses_dir: Directory containing frame_*.png files
        target_length: Target number of frames (default 57)

    Returns:
        List of PIL Images, exactly target_length frames
    """
    # Find all frame files and sort
    frame_files = sorted(poses_dir.glob("frame_*.png"))

    if not frame_files:
        raise ValueError(f"No pose frames found in {poses_dir}")

    frames = []
    for frame_file in frame_files:
        frames.append(Image.open(frame_file).convert("RGB"))

    # Truncate if too many frames
    if len(frames) > target_length:
        frames = frames[:target_length]

    # Pad with last frame if needed
    if len(frames) < target_length:
        last_frame = frames[-1] if frames else None
        if last_frame is None:
            raise ValueError(f"No frames to pad in {poses_dir}")
        num_pad = target_length - len(frames)
        frames.extend([last_frame.copy() for _ in range(num_pad)])

    return frames


def save_latents_and_meta(
    latents: Tensor,
    out_dir: str,
    base_name: str,
    clip_idx: int,
    start_f: int,
    end_f: int,
    fps: float,
    vae_per_channel_normalize: bool,
    is_reference: bool = False,
):
    """Save latents and metadata matching save_vae_latents.py format."""
    os.makedirs(out_dir, exist_ok=True)

    suffix = "_ref" if is_reference else ""
    lat_path = os.path.join(out_dir, f"{base_name}_{clip_idx}{suffix}.pt")
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
    # Only add is_reference field for reference images
    if is_reference:
        meta["is_reference"] = True

    meta_path = os.path.join(out_dir, f"{base_name}_{clip_idx}{suffix}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_vae(ckpt_path: str, device: torch.device) -> CausalVideoAutoencoder:
    """Load VAE model same as save_vae_latents.py."""
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    vae = vae.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    vae.eval().to(device)
    return vae


def process_conditions_folder(
    conditions_dir: Path,
    output_dir: Path,
    vae: CausalVideoAutoencoder,
    height: int,
    width: int,
    clip_length: int,
    vae_per_channel_normalize: bool,
):
    """
    Process all conditioning data from conditions folder.

    For each clip:
    1. Load pose frames, pad to clip_length if needed, encode
    2. Load reference image, encode as single frame
    """
    # Find all JSON metadata files
    json_files = sorted(conditions_dir.glob("*_*.json"))

    # Filter out _ref.json files if any exist
    json_files = [f for f in json_files if not f.name.endswith("_ref.json")]

    print(f"Found {len(json_files)} clip(s) to process")

    for json_file in json_files:
        # Parse base_name and clip_idx from filename
        # Format: {base_name}_{clip_idx}.json
        stem = json_file.stem
        parts = stem.rsplit("_", 1)

        base_name, clip_idx_str = parts
        clip_idx = int(clip_idx_str)

        print(f"\nProcessing: {base_name}_{clip_idx}")

        # Load metadata
        with open(json_file, "r") as f:
            meta = json.load(f)

        fps = meta.get("fps", 25.0)
        start_f = meta.get("start_frame", 0)
        end_f = meta.get("end_frame_exclusive", clip_length)
        poses_dir_name = meta.get("pose_frames_dir", f"{base_name}_{clip_idx}_poses")
        poses_dir = conditions_dir / poses_dir_name

        # Process pose frames
        pose_frames = load_pose_frames(poses_dir, target_length=clip_length)
        print(f"  Loaded {len(pose_frames)} pose frames")

        # Preprocess and encode
        x_pose = preprocess_frames(pose_frames, height, width)
        x_pose = x_pose.to(device=vae.device, dtype=vae.dtype)

        with torch.no_grad():
            lat_pose = vae_encode(
                x_pose,
                vae,
                split_size=1,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )

        # Save pose latents
        save_latents_and_meta(
            latents=lat_pose,
            out_dir=str(output_dir),
            base_name=base_name,
            clip_idx=clip_idx,
            start_f=start_f,
            end_f=end_f,
            fps=fps,
            vae_per_channel_normalize=vae_per_channel_normalize,
            is_reference=False,
        )
        print(f"  Saved pose latents: {base_name}_{clip_idx}.pt")

        # Process reference image
        ref_img_name = meta.get("reference_image", f"{base_name}_{clip_idx}_ref.png")
        ref_img_path = conditions_dir / ref_img_name

        ref_img = Image.open(ref_img_path).convert("RGB")
        print(f"  Loaded reference image: {ref_img_name}")

        # Preprocess as single frame (repeat to make it a "video" of 1 frame)
        x_ref = preprocess_frames([ref_img], height, width)
        x_ref = x_ref.to(device=vae.device, dtype=vae.dtype)

        with torch.no_grad():
            lat_ref = vae_encode(
                x_ref,
                vae,
                split_size=1,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )

        # Save reference latents
        save_latents_and_meta(
            latents=lat_ref,
            out_dir=str(output_dir),
            base_name=base_name,
            clip_idx=clip_idx,
            start_f=start_f,
            end_f=start_f + 1,  # Single frame
            fps=fps,
            vae_per_channel_normalize=vae_per_channel_normalize,
            is_reference=True,
        )
        print(f"  Saved reference latents: {base_name}_{clip_idx}_ref.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Encode pose frames and reference images from conditions folder."
    )
    parser.add_argument(
        "--conditions_dir",
        type=str,
        required=True,
        help="Directory containing conditioning data (JSON, _ref.png, _poses/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for encoded latents (.pt and .json files)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ltxv-2b-0.9.6-dev-04-25.safetensors",
        help="Path to LTX-Video checkpoint (safetensors or dir)",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=57,
        help="Target number of frames per clip (pose frames will be padded to this)",
    )
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--per_channel_normalize",
        type=bool,
        default=True,
        choices=[True, False],
        help="Use per-channel VAE normalization",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    resolved_ckpt = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=str(args.ckpt),
        repo_type="model",
    )

    print(f"Loading VAE from {resolved_ckpt}...")
    vae = load_vae(resolved_ckpt, device=device)
    print("VAE loaded successfully")

    # Process conditions folder
    conditions_dir = Path(args.conditions_dir)
    output_dir = Path(args.output_dir)

    if not conditions_dir.exists():
        raise ValueError(f"Conditions directory does not exist: {conditions_dir}")

    process_conditions_folder(
        conditions_dir=conditions_dir,
        output_dir=output_dir,
        vae=vae,
        height=args.height,
        width=args.width,
        clip_length=args.clip_length,
        vae_per_channel_normalize=args.per_channel_normalize,
    )

    print(f"\nComplete! Encoded latents saved to: {output_dir}")


if __name__ == "__main__":
    main()
