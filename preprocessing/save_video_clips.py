import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import imageio
from PIL import Image


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


def save_clip_and_meta(
    frames: List[Image.Image],
    out_dir: str,
    base_name: str,
    clip_idx: int,
    start_f: int,
    end_f: int,
    fps: float,
    target_height: int = None,
    target_width: int = None,
):
    os.makedirs(out_dir, exist_ok=True)
    
    # Resize frames if target dimensions are specified
    processed_frames = frames
    if target_height is not None and target_width is not None:
        processed_frames = []
        for frame in frames:
            resized_frame = frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
            processed_frames.append(resized_frame)
    
    # Save video clip
    clip_path = os.path.join(out_dir, f"{base_name}_{clip_idx}.mp4")
    writer = imageio.get_writer(clip_path, fps=fps, codec='libx264')
    for frame in processed_frames:
        writer.append_data(np.array(frame))
    writer.close()
    
    # Save metadata (same format as VAE latents)
    meta = {
        "video": base_name,
        "clip_index": clip_idx,
        "start_frame": int(start_f),
        "end_frame_exclusive": int(end_f),
        "fps": float(fps),
        "start_time_sec": float(start_f / max(fps, 1e-8)),
        "end_time_sec": float(end_f / max(fps, 1e-8)),
        "format": "video_clip",
    }
    with open(os.path.join(out_dir, f"{base_name}_{clip_idx}.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Split videos into clips matching VAE latent extraction.")
    parser.add_argument("--inputs", type=str, nargs="+", help="Video files or a directory (processed recursively if dir)")
    parser.add_argument("--output_dir", type=str, default='../../avatars_data/video_clips')
    parser.add_argument("--clip_length", type=int, default=121)
    parser.add_argument("--stride", type=int, default=121, help="Frames to move between clips")
    parser.add_argument("--height", type=int, default=352, help="Target height for frame resizing (matches VAE processing)")
    parser.add_argument("--width", type=int, default=608, help="Target width for frame resizing (matches VAE processing)")
    args = parser.parse_args()

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
            save_clip_and_meta(
                frames=sub,
                out_dir=args.output_dir,
                base_name=base,
                clip_idx=clip_idx,
                start_f=s,
                end_f=e,
                fps=fps,
                target_height=args.height,
                target_width=args.width,
            )
            clip_idx += 1


if __name__ == "__main__":
    import numpy as np
    main()
