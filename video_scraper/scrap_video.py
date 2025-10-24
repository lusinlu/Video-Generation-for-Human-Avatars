import os
import json
import subprocess
import cv2
import torch
import whisperx
import mediapipe as mp
import torchaudio
import pandas as pd
import random
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Video dataset processing script")

parser.add_argument(
    "--csv_path", type=str, default="avspeech_train.csv", help="Path to the CSV file")
parser.add_argument("--start_row", type=int, default=0, help="Start row index (inclusive)")
parser.add_argument("--end_row", type=int, default=1000, help="End row index (exclusive)")
parser.add_argument("--output_dir", type=Path, default=Path("../videos"), help="Directory to save processed videos")
parser.add_argument("--transcripts_file", type=Path, default=Path("video_transcripts.json"), help="Path to transcripts JSON file")

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.96 Safari/537.36",
]

def random_sleep(min_s=1, max_s=4):
    t = random.uniform(min_s, max_s)
    print(f"Sleeping {t:.2f}s...")
    time.sleep(t)

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def run_yt_dlp(cmd, retries=3, sleep_min=2, sleep_max=5):
    """Run yt-dlp with retry logic and random sleep."""
    for attempt in range(retries):
        print(f"Running yt-dlp (attempt {attempt+1}/{retries})...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Check for bot error
        if "Sign in to confirm you're not a bot" in result.stderr:
            raise Exception("YouTube bot detection triggered - stopping script")
        
        if result.returncode == 0:
            random_sleep(sleep_min, sleep_max)
            return True
        random_sleep(10, 20)  # longer backoff on failure
    return False

# -----------------------------
# 3. Pre-filtering function
# -----------------------------
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def is_one_person_from_start(video_path, num_frames=15, fps=2):
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

def pre_filter_batch(csv_path, start_row, end_row, batch_size=10):
    """Pre-filter videos in batches and return filtered videos for each batch."""
    df = pd.read_csv(csv_path, header=None)
    df_subset = df.iloc[start_row:end_row]
    
    # Load Whisper model for transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisperx.load_model("large-v2", device)
    
    total_rows = len(df_subset)
    current_row = start_row
    
    while current_row < start_row + total_rows:
        batch_end = min(current_row + batch_size, start_row + total_rows)
        batch_df = df_subset.iloc[current_row - start_row:batch_end - start_row]
        
        print(f"\n=== Processing batch: rows {current_row} to {batch_end-1} ===")
        filtered_videos = []
        
        for idx, row in batch_df.iterrows():
            ytid, start, end = row[0], float(row[1]), float(row[2])
            row_number = idx + 1  # 1-based row number
            print(f"Pre-filtering row {row_number} (batch {current_row}-{batch_end-1}): {ytid}")
            
            try:
                preview_path = args.output_dir / f"{ytid}_preview.mp4"
                url = f"https://www.youtube.com/watch?v={ytid}"
                ua = get_random_user_agent()
                cookies_arg = ""
                
                cmd = (
                    f'yt-dlp -f mp4 --quiet --no-warnings --merge-output-format mp4 '
                    f'{cookies_arg} --user-agent "{ua}" '
                    f'--download-sections "*{start}-{start+3}" '
                    f'-o "{preview_path}" "{url}"'
                )
                success = run_yt_dlp(cmd)
                if not success or not preview_path.exists():
                    print(f"  Failed to download preview: {ytid}")
                    continue
                
                # Check 1: One person detection
                if not is_one_person_from_start(preview_path):
                    print(f"  Failed person check: {ytid}")
                    preview_path.unlink(missing_ok=True)
                    continue
                    
                # Check 2: Transcribe and check language
                audio_path = str(preview_path.with_suffix(".wav"))
                subprocess.run(f'ffmpeg -y -i "{preview_path}" -vn -ac 1 -ar 16000 "{audio_path}"', shell=True)
                
                result = whisper_model.transcribe(audio_path)
                
                if result["language"] != "en":
                    print(f"  Failed language check (detected {result['language']}): {ytid}")
                    preview_path.unlink(missing_ok=True)
                    os.remove(audio_path)
                    continue
                    
                segments = result.get("segments", [])
                has_text_from_start = any(
                    seg.get("start", 0) < 1.0 and seg.get("text", "").strip()
                    for seg in segments
                )
                
                if not has_text_from_start:
                    print(f"  Failed text-from-start check: {ytid}")
                    preview_path.unlink(missing_ok=True)
                    os.remove(audio_path)
                    continue
                    
                print(f"  Passed all checks: {ytid}")
                filtered_videos.append((ytid, start, end))
                
                preview_path.unlink(missing_ok=True)
                os.remove(audio_path)
                
            except Exception as e:
                if "YouTube bot detection triggered" in str(e):
                    print(f"\n BOT DETECTION ERROR at row {row_number} (video {ytid})")
                    print(f"Last processed row: {row_number}")
                    raise e
                else:
                    print(f"  Error during processing: {ytid} - {e}")
                    preview_path.unlink(missing_ok=True)
                    audio_path = str(preview_path.with_suffix(".wav"))
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    continue
        
        print(f"Batch complete: {len(filtered_videos)}/{len(batch_df)} videos passed")
        yield filtered_videos, current_row, batch_end - 1
        
        current_row = batch_end

# -----------------------------
# 4. Download and trim AVSpeech clips
# -----------------------------
def download_avspeech_subset(filtered_videos, batch_info=""):
    """Download and trim videos with bot error handling."""
    video_paths = []

    for ytid, start, end in filtered_videos:
        url = f"https://www.youtube.com/watch?v={ytid}"
        tmp_path = args.output_dir / f"{ytid}.full.mp4"
        final_path = args.output_dir / f"{ytid}_{int(start*1000)}_{int(end*1000)}.mp4"

        if final_path.exists():
            video_paths.append(final_path)
            continue

        try:
            ua = get_random_user_agent()
            cookies_arg = ""

            if not tmp_path.exists():
                cmd = (
                    f'yt-dlp -f mp4 --quiet --no-warnings --merge-output-format mp4 '
                    f'{cookies_arg} --user-agent "{ua}" '
                    f'-o "{tmp_path}" "{url}"'
                )
                success = run_yt_dlp(cmd)
                if not success or not tmp_path.exists():
                    print(f"Failed to download: {url}")
                    continue

            subprocess.run(
                f'ffmpeg -y -ss {start} -to {end} -i "{tmp_path}" '
                f'-c:v libx264 -preset veryfast -crf 23 -c:a aac "{final_path}"',
                shell=True
            )

            if final_path.exists():
                video_paths.append(final_path)
                if tmp_path.exists():
                    tmp_path.unlink()
                    print(f"Cleaned up temporary file: {tmp_path}")
            else:
                print(f"Failed to cut segment for: {ytid}")
                if tmp_path.exists():
                    tmp_path.unlink()
                    
        except Exception as e:
            if "YouTube bot detection triggered" in str(e):
                print(f"\nBOT DETECTION ERROR during full download of {ytid}")
                print(f"Batch info: {batch_info}")
                raise e
            else:
                print(f"Error downloading {ytid}: {e}")
                continue

    return video_paths

# -----------------------------
# 5. Transcribe audio (WhisperX)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisperx.load_model("large-v2", device)

def transcribe_video(video_path):
    audio_path = str(video_path.with_suffix(".wav"))
    subprocess.run(f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 "{audio_path}"', shell=True)
    result = whisper_model.transcribe(audio_path)
    if result["language"] != "en":
        print(f"Skipping {video_path}, detected language={result['language']}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None
    waveform, sr = torchaudio.load(audio_path)
    align_model, align_metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(
        result["segments"],
        audio=waveform,
        model=align_model,
        device=device,
        align_model_metadata=align_metadata)
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Cleaned up WAV file: {audio_path}")
    
    return aligned_result

# -----------------------------
# 6. Main pipeline with batch processing
# -----------------------------
def process_batch_pipeline():
    """Process videos in batches: preview 10, download full videos, transcribe, then next 10."""
    all_data = []
    existing_paths = set()
    # Load and extend from existing transcripts file if present
    if args.transcripts_file.exists():
        try:
            with open(args.transcripts_file, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    all_data = existing
                    existing_paths = {
                        item.get("video_path")
                        for item in all_data
                        if isinstance(item, dict) and item.get("video_path")
                    }
                    print(f"Loaded {len(all_data)} existing transcripts from {args.transcripts_file}")
        except Exception as e:
            print(f"Warning: could not read existing transcripts: {e}")
    batch_count = 0
    
    try:
        for filtered_videos, batch_start, batch_end in pre_filter_batch(args.csv_path, args.start_row
, args.end_row, batch_size=10):
            batch_count += 1
            batch_info = f"Batch {batch_count} (rows {batch_start}-{batch_end})"
            
            if not filtered_videos:
                print(f"No videos passed filters in {batch_info}")
                continue
                
            print(f"\n=== {batch_info}: Processing {len(filtered_videos)} filtered videos ===")
            
            # Download full videos for this batch
            print(f"Downloading full videos for {batch_info}...")
            video_paths = download_avspeech_subset(filtered_videos, batch_info)
            
            if not video_paths:
                print(f"No videos successfully downloaded in {batch_info}")
                continue
                
            # Transcribe videos for this batch
            print(f"Transcribing {len(video_paths)} videos for {batch_info}...")
            batch_data = []
            
            for i, vp in enumerate(video_paths):
                print(f"Transcribing {i+1}/{len(video_paths)}: {vp}")
                transcript_data = transcribe_video(vp)
                if transcript_data is None:
                    continue
                vp_str = str(vp)
                if vp_str in existing_paths:
                    print(f"Skipping already-recorded transcript for {vp_str}")
                    continue
                batch_data.append({
                    "video_path": vp_str,
                    "transcript": transcript_data["segments"],
                    "batch_info": batch_info
                })
            
            # Add batch data to all data
            all_data.extend(batch_data)
            # Track newly added paths to avoid duplicates in later batches
            for item in batch_data:
                path_val = item.get("video_path")
                if path_val:
                    existing_paths.add(path_val)
            
            # Save progress after each batch
            with open(args.transcripts_file, "w") as f:
                json.dump(all_data, f, indent=2)
            
            print(f" {batch_info} complete: {len(batch_data)} videos transcribed")
            print(f"Total progress: {len(all_data)} videos processed so far")
            
            # Add delay between batches to be gentle on YouTube
            if batch_count < (args.end_row - args.start_row) // 10:
                print("Waiting before next batch...")
                random_sleep(30, 60)  # 30-60 second delay between batches
                
    except Exception as e:
        if "YouTube bot detection triggered" in str(e):
            print("\n SCRIPT STOPPED DUE TO BOT DETECTION")
            print(f"Processed {len(all_data)} videos before stopping")
            print(f"Last completed batch: {batch_count}")
        else:
            print(f"Unexpected error: {e}")
            raise e
    
    # Final save
    with open(args.transcripts_file, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Total videos processed: {len(all_data)}")
    return all_data

# Run the batch pipeline
if __name__ == "__main__":
    process_batch_pipeline()
