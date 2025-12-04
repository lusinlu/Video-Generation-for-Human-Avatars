"""
Utilities for generating FaceFormer-driven 2D face frames from input text.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TTS.api import TTS
from preprocessing.FaceFormer.faceformer import Faceformer


FACEFORMER_DEFAULT_CHECKPOINT = Path("../preprocessing/FaceFormer/vocaset.pth")
FACEFORMER_DEFAULT_TEMPLATE = Path("../preprocessing/FLAME_template.npy")


def _synthesize_tts(text: str, model_name: str, out_wav: Path) -> None:
    if TTS is None:
        raise ImportError(
            "Coqui TTS is not installed. Please install `TTS` to enable automatic narration."
        )
    tts = TTS(model_name=model_name)
    audio = tts.tts(text)
    src_sr = (
        getattr(tts, "speakers_sample_rate", None)
        or getattr(tts, "sample_rate", None)
        or 22050
    )
    wav = np.asarray(audio, dtype=np.float32)
    wav16 = librosa.resample(wav, orig_sr=int(src_sr), target_sr=16000)
    sf.write(out_wav, wav16, 16000, subtype="PCM_16")


def _ensure_audio(
    text: str,
    tts_model: str,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    tmp_dir = tempfile.TemporaryDirectory()
    wav_path = Path(tmp_dir.name) / "tts_16k.wav"
    _synthesize_tts(text, tts_model, wav_path)
    return wav_path, tmp_dir


def _load_audio_tensor(wav_path: Path, device: torch.device) -> torch.Tensor:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000
        )
    audio = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return audio.to(device)


def _load_numpy(path: Path, preferred_keys: Sequence[str] = ("arr_0",)) -> np.ndarray:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path)
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        raise KeyError(f"None of the keys {preferred_keys} found in {path}")
    return np.load(path)


def _load_template_vertices(path: Path) -> np.ndarray:
    verts = _load_numpy(path, preferred_keys=("template", "verts", "vertices", "arr_0"))
    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Template vertices must have shape (N, 3), got {verts.shape}")
    return verts


def _load_faceformer(
    ckpt_path: Path,
    dataset: str,
    feature_dim: int,
    period: int,
    subjects: str,
    vertice_dim: int,
    device: torch.device,
) -> Faceformer:
    if Faceformer is None:
        raise ImportError(
            "FaceFormer package not found. Please ensure `preprocessing/FaceFormer` is installed."
        )
    model = Faceformer(
        dataset=dataset,
        feature_dim=feature_dim,
        vertice_dim=vertice_dim,
        period=period,
        train_subjects=subjects,
        device=device,
    )
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[FaceFormer] Missing weights: {missing}")
    if unexpected:
        print(f"[FaceFormer] Unexpected weights: {unexpected}")
    model.to(device)
    model.eval()
    return model


def _project_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = vertices[:, :2]
    coords = coords - coords.mean(axis=0, keepdims=True)
    max_extent = np.max(np.abs(coords)) + 1e-6
    coords = coords / max_extent
    depth = vertices[:, 2]
    depth = depth - depth.min()
    if depth.max() > 0:
        depth = depth / depth.max()
    return coords, depth


def _render_frame(
    vertices: np.ndarray,
    out_path: Path,
    image_size: int,
    cmap: str,
    face_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Render frame as binary mask (black background, white face dots).

    If face_bbox is provided (x_min, y_min, x_max, y_max) in [0,1] normalized coords,
    positions and scales the face to match the bbox within the image_size canvas.
    """
    coords, depth = _project_vertices(vertices)

    # Create figure with black background
    fig = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_aspect("equal")

    # Set black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if face_bbox is not None:
        # Unpack normalized bbox coordinates
        x_min, y_min, x_max, y_max = face_bbox

        # Convert normalized coordinates to axis coordinates
        # In normalized coords: (0,0) is top-left, (1,1) is bottom-right
        # In matplotlib coords: (-1,-1) is bottom-left, (1,1) is top-right
        # So we need to map and flip Y axis

        # Bbox center in normalized coords
        bbox_center_x = (x_min + x_max) / 2.0
        bbox_center_y = (y_min + y_max) / 2.0

        # Bbox size in normalized coords
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Map normalized coords [0,1] to matplotlib coords [-1,1]
        # X: 0->-1, 1->1, so: x_mpl = 2*x_norm - 1
        # Y: 0->1 (top), 1->-1 (bottom), so: y_mpl = 1 - 2*y_norm (flip Y)
        center_x_mpl = 2.0 * bbox_center_x - 1.0
        center_y_mpl = 1.0 - 2.0 * bbox_center_y

        # Scale factor: bbox uses portion of image, so scale coordinates accordingly
        scale_x = bbox_width * 2.0  # Bbox width maps to range of 2 in matplotlib
        scale_y = bbox_height * 2.0

        # Transform coords: scale them to fit bbox, then translate to bbox center
        coords_scaled = coords.copy()
        coords_scaled[:, 0] = coords_scaled[:, 0] * scale_x + center_x_mpl
        coords_scaled[:, 1] = coords_scaled[:, 1] * scale_y + center_y_mpl

        # Set limits to full image
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        # Render as white dots on black background (binary mask)
        ax.scatter(
            coords_scaled[:, 0], coords_scaled[:, 1], c="white", s=5, linewidths=0
        )
    else:
        # Original behavior: centered face
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        # Render as white dots on black background (binary mask)
        ax.scatter(coords[:, 0], coords[:, 1], c="white", s=5, linewidths=0)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


def _make_one_hot(num_classes: int, index: int, device: torch.device) -> torch.Tensor:
    vec = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
    vec[0, max(0, min(num_classes - 1, index))] = 1.0
    return vec


def generate_faceformer_frames(
    text: str,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    face_bbox: Optional[Tuple[float, float, float, float]] = None,
    dataset: str = "vocaset",
    feature_dim: int = 64,
    period: int = 30,
    train_subjects: str = "F2 F3 F4 M3 M4 M5",
    identity_index: int = 0,
    target_fps: int = 20,
    image_size: int = 512,
    cmap: str = "Spectral",
    device: str = "cuda",
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
) -> Path:
    """
    Generate RGB face frames driven by the provided text.

    Args:
        text: Text to convert to speech and drive facial animation
        output_dir: Directory to save rendered frames
        face_bbox: Optional (x_min, y_min, x_max, y_max) normalized [0,1] coordinates
                   to position and scale the face within the frame
        dataset: FaceFormer dataset name
        feature_dim: Audio feature dimension
        period: FaceFormer period parameter
        train_subjects: Training subjects string
        identity_index: Identity index for one-hot encoding
        target_fps: Target frames per second
        image_size: Output image size (square)
        cmap: Colormap (unused in binary mask mode)
        device: Device to run model on
        tts_model: Text-to-speech model name

    Returns:
        Path to directory containing the rendered frames.
    """

    ckpt_path = FACEFORMER_DEFAULT_CHECKPOINT
    template_path = FACEFORMER_DEFAULT_TEMPLATE

    wav_path, tmp_dir = _ensure_audio(text, tts_model)
    template = _load_template_vertices(template_path)
    vertice_dim = template.shape[0] * 3

    model = _load_faceformer(
        ckpt_path=ckpt_path,
        dataset=dataset,
        feature_dim=feature_dim,
        period=period,
        subjects=train_subjects,
        vertice_dim=vertice_dim,
        device=device,
    )

    audio = _load_audio_tensor(wav_path, device)
    template_tensor = torch.from_numpy(template.reshape(1, -1)).to(device)
    one_hot_dim = model.obj_vector.in_features
    one_hot = _make_one_hot(one_hot_dim, identity_index, device)

    with torch.no_grad():
        predictions = model.predict(audio, template_tensor, one_hot)

    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()

    seq = predictions.reshape(predictions.shape[1], template.shape[0], 3)
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate initial number of output frames based on FPS
    if target_fps and target_fps < period:
        # Downsample from model's native FPS (period) to target_fps
        num_output_frames = int(seq.shape[0] * target_fps / period)
    else:
        # Use all available frames
        num_output_frames = seq.shape[0]

    # Adjust to match the required format: (N * 8 + 1)
    # This ensures compatibility with the model's frame requirements
    num_output_frames_adjusted = ((num_output_frames - 2) // 8 + 1) * 8 + 1
    # Ensure we don't exceed available frames
    num_output_frames_adjusted = min(num_output_frames_adjusted, seq.shape[0])

    # Select evenly spaced frames from the sequence
    frame_indices = np.linspace(
        0, seq.shape[0] - 1, num_output_frames_adjusted, dtype=int
    )

    for output_idx, model_idx in enumerate(frame_indices):
        frame_path = frames_dir / f"frame_{output_idx:05d}.png"
        _render_frame(
            seq[model_idx],
            frame_path,
            image_size=image_size,
            cmap=cmap,
            face_bbox=face_bbox,
        )

    return frames_dir
