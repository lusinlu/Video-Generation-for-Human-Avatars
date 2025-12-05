import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
from PIL import Image
from ltx_video.generate_faceformer_frames import generate_faceformer_frames
from ltx_video.utils.torch_utils import detect_face_bbox


def read_video(path: str) -> Tuple[List[Image.Image], float]:
    """Read video and return frames with FPS."""
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))

    frames: List[Image.Image] = []
    for fr in reader:
        frames.append(Image.fromarray(fr).convert("RGB"))
    reader.close()
    return frames, fps


def iter_clips(num_frames: int, clip_length: int, stride: int) -> List[Tuple[int, int]]:
    """Generate clip ranges matching save_vae_latents.py logic."""
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


def load_transcripts(transcript_json: str) -> Dict[str, List[Dict]]:
    """
    Load transcripts from JSON file.
    
    Supports two formats:
    1. List: [{"video_path": "...", "transcript": [...]}, ...]
    2. Dict: {"video_name": [{"start": ..., "end": ..., "text": ...}], ...}
    
    Returns: Dict mapping video_basename -> transcript segments with words
    """
    with open(transcript_json, "r") as f:
        raw = json.load(f)

    def _basename_no_ext(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    norm: Dict[str, List[Dict]] = {}

    # Handle list format: [{"video_path": "...", "transcript": [...]}, ...]
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
                
            # Extract video path
            video_path = item.get("video_path") or item.get("video") or item.get("file")
            if not video_path:
                continue
                
            base = _basename_no_ext(str(video_path))
            
            # Extract transcript segments
            transcript_segs = item.get("transcript") or item.get("segments") or []
            if not isinstance(transcript_segs, list):
                continue
            
            # Store segments with word-level data preserved
            norm[base] = transcript_segs
    
    # Handle dict format: {"video_name": [...], ...}
    elif isinstance(raw, dict):
        for k, segs in raw.items():
            base = _basename_no_ext(str(k))
            if isinstance(segs, list):
                norm[base] = segs
    return norm


def get_clip_text(
    transcripts: Optional[Dict[str, List[Dict]]],
    video_base: str,
    start_time: float,
    end_time: float,
    default_text: str = "",
) -> str:
    """
    Extract text for a specific clip time range from transcripts using word-level timestamps.

    Uses the 'words' field with individual word timestamps to extract exactly
    the words spoken during the clip's time range.

    Args:
        transcripts: Dictionary of video transcripts with word-level timing
        video_base: Video basename
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        default_text: Default text if no transcript available

    Returns:
        Extracted text for the exact clip time range
    """
    if transcripts is None or video_base not in transcripts:
        return default_text

    segments = transcripts[video_base]
    clip_words = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Skip segments that don't overlap with clip time range
        if seg_start >= end_time or seg_end <= start_time:
            continue

        for word_info in seg["words"]:
            word_start = word_info.get("start", seg_start)
            word_end = word_info.get("end", seg_end)
            word_text = word_info.get("word", "")

            if word_start < end_time and word_end > start_time:
                clip_words.append(word_text)

    result = " ".join(clip_words).strip()
       
    return result if result else default_text


def save_conditioning_data(
    reference_image: Image.Image,
    face_bbox: Tuple[float, float, float, float],
    pose_frames_dir: Path,
    out_dir: str,
    base_name: str,
    clip_idx: int,
    start_f: int,
    end_f: int,
    fps: float,
    text: str,
    num_pose_frames: int,
):
    """
    Save conditioning data for a clip:
    - Reference image (first frame)
    - Face bbox metadata
    - Pose frames directory
    - Metadata JSON
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save reference image
    ref_img_path = os.path.join(out_dir, f"{base_name}_{clip_idx}_ref.png")
    reference_image.save(ref_img_path)

    # Create metadata matching save_vae_latents.py format
    meta = {
        "video": base_name,
        "clip_index": clip_idx,
        "start_frame": int(start_f),
        "end_frame_exclusive": int(end_f),
        "fps": float(fps),
        "start_time_sec": float(start_f / max(fps, 1e-8)),
        "end_time_sec": float(end_f / max(fps, 1e-8)),
        "reference_image": os.path.basename(ref_img_path),
        "face_bbox": {
            "x_min": float(face_bbox[0]),
            "y_min": float(face_bbox[1]),
            "x_max": float(face_bbox[2]),
            "y_max": float(face_bbox[3]),
        },
        "pose_frames_dir": os.path.basename(pose_frames_dir),
        "num_pose_frames": int(num_pose_frames),
        "text": text,
        "format": "conditioning_data",
    }

    meta_path = os.path.join(out_dir, f"{base_name}_{clip_idx}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract conditioning data (reference images + pose frames) from videos."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Video files or a directory (processed recursively if dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for conditioning data",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=57,
        help="Number of frames per clip (must match save_vae_latents.py)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=57,
        help="Frames to move between clips (must match save_vae_latents.py)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=192,
        help="Target height for reference images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target width for reference images",
    )
    parser.add_argument(
        "--transcript_json",
        type=str,
        default=None,
        help="Optional JSON file with transcripts for text conditioning",
    )
    parser.add_argument(
        "--default_text",
        type=str,
        default="",
        help="Default text to use when no transcript is available",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=20,
        help="Target FPS for FaceFormer pose generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for FaceFormer inference (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load transcripts if provided
    transcripts = None
    if args.transcript_json:
        print(f"Loading transcripts from {args.transcript_json}...")
        transcripts = load_transcripts(args.transcript_json)
        print(f"Loaded transcripts for {len(transcripts)} videos")

    # Collect video files
    files: List[str] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi"):
                files.extend([str(pp) for pp in p.rglob(ext)])
        else:
            files.append(str(p))

    print(f"Found {len(files)} video file(s)")

    total_clips = 0
    for vid_idx, vid_path in enumerate(files):
        print(f"\n[{vid_idx + 1}/{len(files)}] Processing: {vid_path}")

        # Read video
        frames, fps = read_video(vid_path)
        if not frames:
            print("No frames found, skipping")
            continue

        base = os.path.splitext(os.path.basename(vid_path))[0]
        clips = iter_clips(len(frames), args.clip_length, args.stride)

        if not clips:
            print("No clips generated, skipping")
            continue

        print(f"  Found {len(clips)} clip(s)")

        clip_idx = 0
        for s, e in clips:
            # Extract first frame as reference
            first_frame = frames[s]

            # Resize to target dimensions
            reference_image = first_frame.resize(
                (args.width, args.height), Image.BICUBIC
            )
            try:
                face_bbox = detect_face_bbox(reference_image)
            except Exception as ex:
                print(f" Clip {clip_idx}: Face detection failed - {ex}")
                clip_idx += 1
                continue

            # Get text for this clip using word-level timestamps
            start_time = s / max(fps, 1e-8)
            end_time = e / max(fps, 1e-8)

            clip_text = get_clip_text(
                transcripts, base, start_time, end_time, args.default_text
            )

            # Use default text if no text found
            # FaceFormer needs some text to generate audio, so use a neutral phrase
            if not clip_text.strip():
                clip_text = (
                    args.default_text if args.default_text.strip() else "neutral"
                )

            print(
                f"  Clip {clip_idx}: [{start_time:.2f}s-{end_time:.2f}s] "
                f"text='{clip_text[:60]}...' ({len(clip_text)} chars)"
            )

            # Generate FaceFormer pose frames
            # Number of frames should match clip length exactly
            num_frames = e - s

            pose_frames_dir = Path(args.output_dir) / f"{base}_{clip_idx}_poses"

            generated_dir = generate_faceformer_frames(
                text=clip_text,
                output_dir=str(pose_frames_dir),
                face_bbox=face_bbox,
                num_frames=num_frames,  # Specify exact number of frames to match clip
                target_fps=args.target_fps,
                height=args.height,
                width=args.width,
                device=args.device,
            )

            # Count generated frames
            num_pose_frames = len(
                [f for f in os.listdir(generated_dir) if f.lower().endswith(".png")]
            )

            # Verify frame count matches expected
            if num_pose_frames != num_frames:
                print(
                    f"  ⚠ Clip {clip_idx}: Expected {num_frames} pose frames, got {num_pose_frames}"
                )

            # Save conditioning data
            save_conditioning_data(
                reference_image=reference_image,
                face_bbox=face_bbox,
                pose_frames_dir=pose_frames_dir,
                out_dir=args.output_dir,
                base_name=base,
                clip_idx=clip_idx,
                start_f=s,
                end_f=e,
                fps=fps,
                text=clip_text,
                num_pose_frames=num_pose_frames,
            )

            print(
                f" Clip {clip_idx}: Saved reference + {num_pose_frames} pose frames"
            )
            clip_idx += 1
            total_clips += 1

    print(f"Complete! Processed {total_clips} clips from {len(files)} video(s)")
    print(
        f"\nOutput directory: {args.output_dir}"
        f"\n  - Reference images: {base}_{{clip_idx}}_ref.png"
        f"\n  - Pose frames: {base}_{{clip_idx}}_poses/"
        f"\n  - Metadata: {base}_{{clip_idx}}.json"
        f"\n\nNote: Ensure this matches the number of clips from save_vae_latents.py"
        f"\n      (same --clip_length and --stride parameters)"
    )


if __name__ == "__main__":
    main()
