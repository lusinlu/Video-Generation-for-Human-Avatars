import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
from FaceFormer.faceformer import Faceformer
from TTS.api import TTS
from transformers import Wav2Vec2Processor


def _basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def load_transcripts(transcript_json: str) -> Dict[str, List[Dict]]:
    """
    Normalize transcripts to mapping: video_basename -> [{start, end, text}, ...]
    """
    with open(transcript_json, "r") as f:
        raw = json.load(f)

    def _get_time(seg, key_candidates):
        for k in key_candidates:
            if k in seg:
                try:
                    return float(seg[k])
                except Exception:
                    pass
        return None

    def _get_text(seg):
        for k in ("text", "caption", "utterance", "transcript"):
            if k in seg and str(seg[k]).strip():
                return str(seg[k]).strip()
        return ""

    def _get_words(seg):
        words = seg.get("words")
        if isinstance(words, list):
            items = []
            for w in words:
                try:
                    items.append(
                        {
                            "word": str(w.get("word", "")).strip(),
                            "start": float(w.get("start", 0.0)),
                            "end": float(w.get("end", 0.0)),
                        }
                    )
                except Exception:
                    continue
            return items
        return None

    norm: Dict[str, List[Dict]] = {}

    if isinstance(raw, dict):
        for k, segs in raw.items():
            base = _basename_no_ext(str(k))
            lst: List[Dict] = []
            if isinstance(segs, list):
                for seg in segs:
                    s = _get_time(seg, ("start", "start_time", "from"))
                    e = _get_time(seg, ("end", "end_time", "to"))
                    t = _get_text(seg)
                    if s is None or e is None:
                        continue
                    lst.append({"start": float(s), "end": float(e), "text": t})
            norm[base] = sorted(lst, key=lambda d: d.get("start", 0.0))
    elif isinstance(raw, list):
        for item in raw:
            vid_path = (
                item.get("video")
                or item.get("file")
                or item.get("path")
                or item.get("video_path")
            )
            if "transcript" in item and isinstance(item["transcript"], list):
                base = _basename_no_ext(str(vid_path)) if vid_path else None
                if not base:
                    continue
                for seg in item["transcript"]:
                    s = _get_time(seg, ("start", "start_time", "from"))
                    e = _get_time(seg, ("end", "end_time", "to"))
                    t = _get_text(seg)
                    w = _get_words(seg)
                    if s is None or e is None:
                        continue
                    norm.setdefault(base, []).append(
                        {"start": float(s), "end": float(e), "text": t, "words": w}
                    )
            elif "segments" in item and isinstance(item["segments"], list):
                base = _basename_no_ext(str(vid_path)) if vid_path else None
                if not base:
                    continue
                for seg in item["segments"]:
                    s = _get_time(seg, ("start", "start_time", "from"))
                    e = _get_time(seg, ("end", "end_time", "to"))
                    t = _get_text(seg)
                    w = _get_words(seg)
                    if s is None or e is None:
                        continue
                    norm.setdefault(base, []).append(
                        {"start": float(s), "end": float(e), "text": t, "words": w}
                    )
            else:
                base = _basename_no_ext(str(vid_path)) if vid_path else None
                if not base:
                    continue
                s = _get_time(item, ("start", "start_time", "from"))
                e = _get_time(item, ("end", "end_time", "to"))
                t = _get_text(item)
                if s is None or e is None:
                    continue
                norm.setdefault(base, []).append(
                    {"start": float(s), "end": float(e), "text": t}
                )
        for k in list(norm.keys()):
            norm[k] = sorted(norm[k], key=lambda d: d.get("start", 0.0))
    else:
        raise ValueError("Unsupported transcripts JSON format.")

    return norm


def load_clip_metas(latents_dir: str) -> List[Tuple[str, Dict]]:
    metas: List[Tuple[str, Dict]] = []
    for p in sorted(Path(latents_dir).glob("*.json")):
        with open(p, "r") as f:
            meta = json.load(f)
        metas.append((p.stem, meta))  # (videoBase_i, meta)
    return metas


def collect_text_for_window(
    trans_map: Dict[str, List[Dict]], video_base: str, t0: float, t1: float
) -> str:
    segments = trans_map.get(video_base)
    if segments is None:
        segments = trans_map.get(video_base.lower(), [])
    words_accum: List[str] = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if e <= t0 or s >= t1:
            continue
        seg_words = seg.get("words")
        for w in seg_words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", 0.0))
            if we <= t0 or ws >= t1:
                continue
            token = str(w.get("word", "")).strip()
            if token:
                words_accum.append(token)
    return " ".join(words_accum).strip()


def _strip_index_suffix(name: str) -> str:
    if "_" in name:
        parts = name.rsplit("_", 1)
        if parts[1].isdigit():
            return parts[0]
    return name


def synthesize_audio(text: str, model_name: str, out_wav_16k: str) -> None:

    tts = TTS(model_name=model_name)
    wav = tts.tts(text)

    src_sr = getattr(tts, "speakers_sample_rate", None)
    if src_sr is None:
        src_sr = 22050

    y = np.asarray(wav, dtype=np.float32)
    y16 = librosa.resample(y, orig_sr=int(src_sr), target_sr=16000)
    sf.write(out_wav_16k, y16, 16000, subtype="PCM_16")


def extract_faceformer_latents(model: Faceformer, wav_path: str) -> np.ndarray:
    """
    Convert 16kHz mono wav to FaceFormer latent features via a provided API.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000
        )
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    input_values = np.squeeze(processor(audio, sampling_rate=16000).input_values)
    input_values = np.reshape(input_values, (-1, input_values.shape[0]))  # (1, T)
    device = next(model.parameters()).device
    x = torch.FloatTensor(input_values).to(device)
    with torch.no_grad():
        lat = model.extract_audio_motion_features(x)
    if torch.is_tensor(lat):
        lat = lat.detach().cpu().numpy()
    return np.asarray(lat)


def main():
    parser = argparse.ArgumentParser(
        description="Align clip texts → TTS audio → FaceFormer latents"
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory with clip metadata JSONs from save_vae_latents.py",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        help="Transcript JSON with per-video segments",
    )
    parser.add_argument(
        "--audio_out",
        type=str,
        required=True,
        help="Directory to save generated 16kHz wavs",
    )
    parser.add_argument(
        "--ff_out", type=str, required=True, help="Directory to save FaceFormer latents"
    )
    parser.add_argument("--tts_model", type=str, default="tts_models/en/ljspeech/vits")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    os.makedirs(args.audio_out, exist_ok=True)
    os.makedirs(args.ff_out, exist_ok=True)

    trans_map = load_transcripts(args.transcripts)
    metas = load_clip_metas(args.latents_dir)

    # Load FaceFormer
    model = Faceformer(device=args.device)
    ckpt_path = Path("FaceFormer") / "vocaset.pth"
    sd = torch.load(str(ckpt_path))
    model.load_state_dict(sd, strict=False)
    model = model.to(args.device).eval()
    for stem, meta in metas:
        video_base = str(meta["video"])
        stem_base = _strip_index_suffix(stem)
        t0 = (
            float(meta["start_time_sec"])
            if "start_time_sec" in meta
            else float(meta["start_frame"]) / float(meta["fps"])
        )
        t1 = (
            float(meta["end_time_sec"])
            if "end_time_sec" in meta
            else float(meta["end_frame_exclusive"]) / float(meta["fps"])
        )
        text = collect_text_for_window(trans_map, video_base, t0, t1)
        if not text and stem_base != video_base:
            text = collect_text_for_window(trans_map, stem_base, t0, t1)
        if not text:
            continue

        # Synthesize audio
        wav_path = os.path.join(args.audio_out, f"{stem}.wav")
        synthesize_audio(text, args.tts_model, wav_path)

        # Extract FaceFormer latents and save
        lat = extract_faceformer_latents(model, wav_path)
        out_npy = os.path.join(args.ff_out, f"{stem}_ff.npy")
        np.save(out_npy, lat)
        # Save sidecar with text and timing
        with open(os.path.join(args.ff_out, f"{stem}_text.json"), "w") as f:
            json.dump(
                {
                    "text": text,
                    "start_time_sec": t0,
                    "end_time_sec": t1,
                    "wav": wav_path,
                    "faceformer_latents": out_npy,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
