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


# 51 Static landmark vertex indices (landmarks 17-67)
# Using closest vertex to barycentric position
FLAME_51_STATIC_LANDMARK_INDICES = np.array(
    [
        # Right eyebrow: indices 0-4 (landmarks 17-21)
        3763,
        2566,
        335,
        3154,
        3712,
        # Left eyebrow: indices 5-9 (landmarks 22-26)
        3868,
        2135,
        16,
        17,
        3892,
        # Nose: indices 10-18 (landmarks 27-35)
        # 3560, 3561, 3508, 3564, 2748, 2792, 3556, 1675, 1612,
        # Right eye: indices 19-24 (landmarks 36-41)
        2437,
        2383,
        2494,
        3632,
        2293,
        2296,
        # Left eye: indices 25-30 (landmarks 42-47)
        3833,
        1343,
        1034,
        1175,
        884,
        881,
        # Mouth outer: indices 31-42 (landmarks 48-59)
        2715,
        2813,
        2774,
        3543,
        1657,
        1696,
        1579,
        1795,
        1865,
        3503,
        2948,
        2898,
        # Mouth inner: indices 43-50 (landmarks 60-67)
        2845,
        2785,
        3533,
        1668,
        1730,
        1848,
        3509,
        2937,
    ],
    dtype=np.int64,
)


FLAME_68_LANDMARK_INDICES = FLAME_51_STATIC_LANDMARK_INDICES


def _get_facial_feature_indices_landmarks(vertices: np.ndarray) -> np.ndarray:
    """
    Get the vertex indices for the 68 facial landmarks in FLAME mesh.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N,3), got {vertices.shape}")

    # Filter to valid indices within the vertex array
    valid_indices = FLAME_68_LANDMARK_INDICES[FLAME_68_LANDMARK_INDICES < len(vertices)]

    return valid_indices


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
    height: int,
    width: int,
    cmap: str,
    face_bbox: Optional[Tuple[float, float, float, float]] = None,
    features_only: bool = True,
    point_size: float = 5.0,
) -> None:
    """
    Args:
        vertices: Face mesh vertices (N, 3)
        out_path: Path to save the rendered image
        height: Output image height in pixels
        width: Output image width in pixels
        cmap: Colormap (unused in binary mask mode)
        face_bbox: Optional (x_min, y_min, x_max, y_max) normalized [0,1] coordinates
                   to position and scale the face within the frame
        features_only: If True, only render facial features (mouth, eyes, eyebrows, nose).
                      If False, render all vertices.
        point_size: Size of rendered dots in points² (matplotlib scatter 's' parameter).
    """
    # Select vertices to render
    if features_only:
        # Select only facial feature vertices using landmark-based selection
        feature_indices = _get_facial_feature_indices_landmarks(vertices)
        if len(feature_indices) == 0:
            raise ValueError("No landmark vertices found")
        vertices_to_render = vertices[feature_indices]

    else:
        vertices_to_render = vertices

    coords, depth = _project_vertices(vertices_to_render)

    # Create figure with exact dimensions (no padding)
    # figsize is in inches, dpi converts to pixels: width_pixels = width_inches * dpi
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Use add_axes to remove all margins
    ax.axis("off")

    # Set black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Unpack normalized bbox coordinates [0,1]
    x_min, y_min, x_max, y_max = face_bbox

    # Bbox center and size in normalized coords
    bbox_center_x = (x_min + x_max) / 2.0
    bbox_center_y = (y_min + y_max) / 2.0
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Convert to pixel coordinates
    center_x_px = bbox_center_x * width
    center_y_px = bbox_center_y * height
    bbox_width_px = bbox_width * width
    bbox_height_px = bbox_height * height

    # Transform coords from normalized [-1,1] to bbox position in pixels
    coords_transformed = coords.copy()
    coords_transformed[:, 0] = (coords[:, 0] * bbox_width_px / 2.0) + center_x_px
    coords_transformed[:, 1] = (
        -coords[:, 1] * bbox_height_px / 2.0
    ) + center_y_px  # Negative Y to flip

    # Set limits to image dimensions (pixel coordinates)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Render as white dots
    ax.scatter(
        coords_transformed[:, 0],
        coords_transformed[:, 1],
        c="white",
        s=point_size,
        linewidths=0,
    )

    # Save with exact dimensions (no bbox_inches='tight' to avoid cropping)
    fig.savefig(out_path, dpi=dpi, facecolor="black", pad_inches=0)
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
    num_frames: Optional[int] = None,
    features_only: bool = True,
    point_size: float = 2.0,
    dataset: str = "vocaset",
    feature_dim: int = 64,
    period: int = 30,
    train_subjects: str = "F2 F3 F4 M3 M4 M5",
    identity_index: int = 0,
    target_fps: int = 20,
    height: int = 512,
    width: int = 512,
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
        num_frames: Optional exact number of frames to generate. If provided,
                    overrides automatic frame count calculation based on audio duration.
        features_only: If True, only render facial features (mouth, eyes, eyebrows, nose).
                      If False, render all face vertices. Default: True
        point_size: Size of rendered dots in points² (matplotlib scatter 's' parameter).
        dataset: FaceFormer dataset name
        feature_dim: Audio feature dimension
        period: FaceFormer period parameter
        train_subjects: Training subjects string
        identity_index: Identity index for one-hot encoding
        target_fps: Target frames per second
        height: Output image height in pixels
        width: Output image width in pixels
        cmap: Colormap (unused in binary mask mode)
        device: Device to run model on
        tts_model: Text-to-speech model name

    Returns:
        Path to directory containing the rendered frames.
    """

    ckpt_path = FACEFORMER_DEFAULT_CHECKPOINT
    template_path = FACEFORMER_DEFAULT_TEMPLATE

    wav_path, tmp_dir = _ensure_audio(text, tts_model)

    audio_data, audio_sr = sf.read(wav_path)

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
    max_audio_length = 600 * 16000 // 30  # ~320,000 samples
    if audio.shape[1] > max_audio_length:
        print(
            f"[FaceFormer] Audio too long ({audio.shape[1]} samples), "
            f"truncating to {max_audio_length} samples (~20s)"
        )
        audio = audio[:, :max_audio_length]

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

    # Determine number of output frames
    if num_frames is not None:
        # Use explicitly specified number of frames
        num_output_frames_adjusted = num_frames
    else:
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
            height=height,
            width=width,
            cmap=cmap,
            face_bbox=face_bbox,
            features_only=features_only,
            point_size=point_size,
        )

    return frames_dir
