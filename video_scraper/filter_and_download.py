import os
import json
import random
import subprocess
import concurrent.futures
import time
from pathlib import Path
from typing import List, Tuple

import argparse
import cv2
import mediapipe as mp
import pandas as pd
import shutil
FFMPEG_PATH = shutil.which("ffmpeg")
print(FFMPEG_PATH)
def random_sleep(min_s=1, max_s=4):
    t = random.uniform(min_s, max_s)
    print(f"Sleeping {t:.2f}s...")
    time.sleep(t)


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.96 Safari/537.36",
]


def get_random_user_agent():
    return random.choice(USER_AGENTS)


def run_yt_dlp(cmd: str, retries: int = 2, sleep_min: int = 2, sleep_max: int = 5, sleep_after_success: bool = True,
) -> bool:
    for attempt in range(retries):
        print(f"Running yt-dlp (attempt {attempt+1}/{retries})...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Check for bot error
        if "Sign in to confirm" in result.stderr:
            raise Exception("YouTube bot detection triggered - stopping script")
        if "Video unavailable. This video" in result.stderr:
            print(f"Video unavailable")
            return False
        if " Private video. Sign" in result.stderr:
            print(f"Private video")
            return False

        if result.returncode == 0:
            if sleep_after_success:
                random_sleep(sleep_min, sleep_max)
            return True
        if result.stderr:
            print("stderr:\n" + result.stderr.strip())
        if result.stdout:
            print("stdout:\n" + result.stdout.strip())
        random_sleep(3, 6)
    return False


mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def is_one_person_from_start(video_path: Path, num_frames: int = 15, fps: int = 2) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps) if frame_rate > 0 else 1

    frames_checked = 0
    for frame_idx in range(0, num_frames * interval, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(frame_rgb)
        if not results.detections:
            cap.release()
            continue
        if len(results.detections) != 1:
            return False
        frames_checked += 1

    cap.release()
    return frames_checked > 1


def pre_filter_batches(
    csv_path: str, start_row: int, end_row: int, output_dir: Path, batch_size: int = 10, workers: int = 4,):
    """Yield filtered (ytid, start, end) per batch along with batch range."""
    df = pd.read_csv(csv_path, header=None)
    if end_row == -1 or end_row is None:
        end_row = len(df)
    df_subset = df.iloc[start_row:end_row]

    total_rows = len(df_subset)
    current_row = start_row

    while current_row < start_row + total_rows:
        batch_end = min(current_row + batch_size, start_row + total_rows)
        batch_df = df_subset.iloc[current_row - start_row : batch_end - start_row]

        print(f"\n=== Pre-filtering rows {current_row} to {batch_end-1} ===")
        batch_filtered: List[Tuple[str, float, float]] = []

        def process_row(idx: int, ytid: str, start: float, end: float):
            row_number_local = idx + 1
            try:
                preview_path = output_dir / f"{ytid}_preview.mp4"
                url = f"https://www.youtube.com/watch?v={ytid}"
                ua = get_random_user_agent()
                cmd = (
                    f'yt-dlp --retries 2 --fragment-retries 2 --socket-timeout 10 '
                    f'--no-progress --quiet --no-warnings '
                    f'-f mp4 --merge-output-format mp4 '
                    f'--ffmpeg-location "{FFMPEG_PATH}" '
                    f'--user-agent "{ua}" '
                    f'--download-sections "*{start}-{start+3}" '
                    f'-o "{preview_path}" "{url}"'
                )
                success = run_yt_dlp(cmd, sleep_after_success=False)
                if not success or not preview_path.exists():
                    print(f"  Failed to download preview: {ytid}")
                    return None

                if not is_one_person_from_start(preview_path):
                    print(f"  Failed person check: {ytid}")
                    preview_path.unlink(missing_ok=True)
                    return None

                preview_path.unlink(missing_ok=True)
                return (ytid, start, end)

            except Exception as e:
                if "YouTube bot detection triggered" in str(e):
                    print(f"\n BOT DETECTION ERROR at row {row_number_local} (video {ytid})")
                    print(f"Last processed row: {row_number_local}")
                    raise e
                print(f"Error during pre-filter for {ytid}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = []
            for idx, row in batch_df.iterrows():
                ytid, start, end = row[0], float(row[1]), float(row[2])
                futures.append(
                    executor.submit(process_row, idx, ytid, start, end)
                )
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is not None:
                    batch_filtered.append(result)

        print(f"Batch complete: {len(batch_filtered)}/{len(batch_df)} passed")
        yield batch_filtered, current_row, batch_end - 1
        current_row = batch_end


def download_avspeech_subset(
    filtered_videos: List[Tuple[str, float, float]],
    output_dir: Path,
    batch_info: str = "",
    workers: int = 4,
) -> List[Path]:
    """Download and trim videos with bot error handling, in parallel."""
    def process_video(ytid: str, start: float, end: float):
        url = f"https://www.youtube.com/watch?v={ytid}"
        tmp_path = output_dir / f"{ytid}.full.mp4"
        final_path = output_dir / f"{ytid}_{int(start*1000)}_{int(end*1000)}.mp4"

        if final_path.exists():
            return final_path

        try:
            ua = get_random_user_agent()
            if not tmp_path.exists():
                cmd = (
                    f'yt-dlp --retries 2 --fragment-retries 2 --socket-timeout 10 '
                    f'--no-progress --quiet --no-warnings '
                    f'-f mp4 --merge-output-format mp4 '
                    f'--ffmpeg-location "{FFMPEG_PATH}" '
                    f'--user-agent "{ua}" '
                    f'-o "{tmp_path}" "{url}"'
                )
                success = run_yt_dlp(cmd, sleep_after_success=True)
                if not success or not tmp_path.exists():
                    print(f"Failed to download: {url}")
                    return None

            subprocess.run(
                f'ffmpeg -hide_banner -loglevel error -nostats -y -ss {start} -to {end} -i "{tmp_path}" '
                f'-c:v libx264 -preset veryfast -crf 23 -c:a aac "{final_path}"',
                shell=True,
            )

            if final_path.exists():
                if tmp_path.exists():
                    tmp_path.unlink()
                    print(f"Cleaned up temporary file: {tmp_path}")
                return final_path
            else:
                print(f"Failed to cut segment for: {ytid}")
                if tmp_path.exists():
                    tmp_path.unlink()
                return None

        except Exception as e:
            if "YouTube bot detection triggered" in str(e):
                print(f"\nBOT DETECTION ERROR during full download of {ytid}")
                print(f"Batch info: {batch_info}")
                raise e
            print(f"Error downloading {ytid}: {e}")
            return None

    video_paths: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(process_video, ytid, float(start), float(end))
            for (ytid, start, end) in filtered_videos
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                if res is not None:
                    video_paths.append(res)
            except Exception as e:
                # If a worker raised bot detection, propagate immediately
                if "YouTube bot detection triggered" in str(e):
                    raise e
                print(f"Worker error: {e}")

    return video_paths


def main():
    parser = argparse.ArgumentParser(description="Prefilter AVSpeech and download trimmed clips")
    parser.add_argument("--csv_path", type=str, default="avspeech_train.csv",)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=-1)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--manifest", type=Path, default=Path("downloaded_videos.json"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if present
    all_records: List[dict] = []
    existing_paths = set()
    if args.manifest.exists():
        try:
            with open(args.manifest, "r") as f:
                all_records = json.load(f) or []
            existing_paths = {r.get("video_path") for r in all_records if isinstance(r, dict)}
            print(f"Loaded {len(all_records)} existing entries from manifest")
        except Exception as e:
            print(f"Warning: could not read manifest: {e}")

    # Process batches: filter → download → append manifest per batch
    for batch_filtered, b_start, b_end in pre_filter_batches(
        csv_path=args.csv_path,
        start_row=args.start_row,
        end_row=args.end_row,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    ):
        if not batch_filtered:
            continue
        print(f"\nDownloading {len(batch_filtered)} filtered items for rows {b_start}-{b_end}...")
        batch_paths = download_avspeech_subset(batch_filtered, args.output_dir)
        new_records = []
        for p in batch_paths:
            p_str = str(p)
            if p_str in existing_paths:
                continue
            new_records.append({"video_path": p_str, "ytid": Path(p).name.split("_")[0]})
            existing_paths.add(p_str)
        if new_records:
            all_records.extend(new_records)
            with open(args.manifest, "w") as f:
                json.dump(all_records, f, indent=2)
            print(f"Appended {len(new_records)} new entries → {args.manifest} (total {len(all_records)})")
        else:
            print("No new entries to append for this batch")


if __name__ == "__main__":
    main()


