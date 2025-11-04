import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import argparse
import torch
import torchaudio
import whisperx


def transcribe_video(video_path: Path, whisper_model) -> Dict:
    audio_path = str(video_path.with_suffix(".wav"))
    subprocess.run(
        f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 "{audio_path}"',
        shell=True,
    )
    result = whisper_model.transcribe(audio_path)
    if result.get("language") != "en":
        print(f"Skipping {video_path}, detected language={result.get('language')}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {}
    waveform, sr = torchaudio.load(audio_path)
    align_device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result.get("language"),
        device=align_device,
    )
    aligned_result = whisperx.align(
        result.get("segments", []),
        audio=waveform,
        model=align_model,
        device=align_device,
        align_model_metadata=align_metadata,
    )
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return aligned_result


def find_first_speech_timestamp(segments: List[Dict]) -> Optional[float]:
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        try:
            return float(seg.get("start", 0.0))
        except Exception:
            continue
    return None


def trim_video_to_start(src: Path, dst: Path, start_sec: float) -> bool:
    """Trim src video to start at start_sec (copy codecs) and write to dst."""
    tmp = dst.with_suffix(".tmp.mp4")
    # Using re-encode for reliability around non-keyframe starts
    cmd = (
        f'ffmpeg -y -ss {start_sec:.3f} -i "{src}" '
        f'-c:v libx264 -preset veryfast -crf 23 -c:a aac "{tmp}"'
    )
    rc = subprocess.run(cmd, shell=True).returncode
    if rc == 0 and tmp.exists():
        tmp.replace(dst)
        return True
    if tmp.exists():
        tmp.unlink()
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Process downloaded videos: transcribe and align"
    )
    parser.add_argument(
        "--videos_dir",
        type=Path,
        required=True,
        help="Directory with downloaded videos",
    )
    parser.add_argument(
        "--transcripts_file",
        type=Path,
        default=Path("video_transcripts.json"),
        help="Path to transcripts JSON file (output)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisperx.load_model("large-v2", device)

    all_data: List[Dict] = []
    if args.transcripts_file.exists():
        try:
            with open(args.transcripts_file, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    all_data = existing
        except Exception:
            pass

    paths = sorted(list(Path(args.videos_dir).glob("*.mp4")))
    for i, vp in enumerate(paths):
        print(f"Transcribing {i+1}/{len(paths)}: {vp}")
        transcript_data = transcribe_video(vp, whisper_model)
        if not transcript_data:
            continue
        # Find first speech timestamp from aligned segments
        first_speech = find_first_speech_timestamp(transcript_data.get("segments", []))
        if first_speech is not None and first_speech > 0.0:
            trimmed = vp.with_name(vp.stem + "_trimmed.mp4")
            if trim_video_to_start(vp, trimmed, first_speech):
                # Re-run transcription on trimmed video to produce aligned JSON for final cut
                print(
                    f"Re-transcribing trimmed video starting at {first_speech:.2f}s: {trimmed}"
                )
                transcript_data = transcribe_video(trimmed, whisper_model)
                if not transcript_data:
                    # Fall back to original if something failed
                    final_video_path = str(vp)
                else:
                    final_video_path = str(trimmed)
            else:
                final_video_path = str(vp)
        else:
            final_video_path = str(vp)

        all_data.append(
            {
                "video_path": final_video_path,
                "transcript": transcript_data.get("segments", []),
            }
        )

        with open(args.transcripts_file, "w") as f:
            json.dump(all_data, f, indent=2)

    print(f"Processed {len(all_data)} videos → {args.transcripts_file}")
    for pattern in ("*_preview.mp4", "*_preview_trimmed.mp4"):
        for p in Path(args.videos_dir).glob(pattern):
            try:
                p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
