import torch
from torch import nn
from safetensors.torch import save_file
import json
import copy
import os
import numpy as np
from typing import Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        super().__init__()

    # pylint: disable=unused-argument
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


def save_module_safetensors(
    module: nn.Module, target_path: str, metadata: dict | None = None
) -> None:
    """Save an nn.Module's state_dict as a single safetensors file (CPU tensors).

    This does not mutate the module. All tensors are moved to CPU before writing.
    """
    state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    meta = dict(metadata or {})
    # If caller didn't provide a 'config', embed a minimal transformer config when available
    if "config" not in meta:
        try:
            cfg = getattr(module, "config", None)
            if cfg is not None:
                cfg_dict = {
                    k: v
                    for k, v in getattr(cfg, "__dict__", {}).items()
                    if not k.startswith("_")
                }
                meta["config"] = json.dumps({"transformer": cfg_dict})
        except Exception:
            pass
    # Safetensors requires metadata values to be strings; coerce defensively
    meta = {str(k): (v if isinstance(v, str) else str(v)) for k, v in meta.items()}
    save_file(state, target_path, metadata=meta)


def export_merged_safetensors(
    peft_model: nn.Module, target_path: str, metadata: dict | None = None
) -> None:
    """Deepcopy a PEFT-wrapped model, merge LoRA into base weights, and save as safetensors.

    - Does NOT modify the original model (safe during training).
    - Requires the model to support `.merge_and_unload()` (PEFT).
    """
    model_copy = copy.deepcopy(peft_model)
    if not hasattr(model_copy, "merge_and_unload"):
        raise AttributeError(
            "Model does not support merge_and_unload(); cannot export merged weights"
        )
    merged = model_copy.merge_and_unload()
    # Build metadata with embedded config for single-file load
    meta = dict(metadata or {})
    try:
        cfg = getattr(merged, "config", None)
        if cfg is not None:
            # Extract plain dict of non-private fields
            cfg_dict = {
                k: v
                for k, v in getattr(cfg, "__dict__", {}).items()
                if not k.startswith("_")
            }
            # Merge optional scheduler config provided by caller
            scheduler_cfg = meta.pop("scheduler", None)
            config_root = {"transformer": cfg_dict}
            if scheduler_cfg is not None:
                config_root["scheduler"] = scheduler_cfg
            meta["config"] = json.dumps(config_root)
    except Exception:
        pass
    # Safetensors requires metadata values to be strings; coerce defensively
    meta = {str(k): (v if isinstance(v, str) else str(v)) for k, v in meta.items()}
    state = {k: v.detach().cpu() for k, v in merged.state_dict().items()}
    save_file(state, target_path, metadata=meta)


def save_training_checkpoint(
    model: nn.Module,
    target_path: str,
    train_mode: str,
    metadata: dict | None = None,
    is_best: bool = False,
) -> None:
    """
    Unified model saving function for training checkpoints.

    Args:
        model: The model to save
        target_path: Path where to save the model
        train_mode: Either "full" or "lora_audio"
        metadata: Optional metadata to include in the saved file
        is_best: If True, adds "best" prefix to filename and includes best_loss in metadata
    """
    # Adjust filename and metadata for best model
    if is_best:
        dir_path = os.path.dirname(target_path)
        filename = os.path.basename(target_path)
        if not filename.startswith("best_"):
            filename = f"best_{filename}"
        target_path = os.path.join(dir_path, filename)

    if train_mode == "lora_audio":
        export_merged_safetensors(model, target_path, metadata)
    else:
        save_module_safetensors(model, target_path, metadata)


def detect_face_bbox(
    image: Union["Image.Image", np.ndarray, torch.Tensor],
    min_detection_confidence: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Detect a single face in the image and return normalized bounding box coordinates.

    Uses MediaPipe Face Detection to locate exactly one face in the input image.

    Args:
        image: Input image as PIL Image, numpy array (H, W, 3), or torch.Tensor (3, H, W) or (H, W, 3)
        min_detection_confidence: Minimum confidence for face detection (0.0 to 1.0)

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
        where (x_min, y_min) is top-left and (x_max, y_max) is bottom-right

    Raises:
        ImportError: If mediapipe is not installed
        ValueError: If no face is detected or multiple faces are detected

    Example:
        >>> from PIL import Image
        >>> image = Image.open("face.jpg")
        >>> x_min, y_min, x_max, y_max = detect_single_face_bbox(image)
        >>> print(f"Face bbox: ({x_min:.3f}, {y_min:.3f}) to ({x_max:.3f}, {y_max:.3f})")
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "MediaPipe is required for face detection. "
            "Install it with: pip install mediapipe"
        )

    # Convert input to numpy array (H, W, 3) in RGB format
    if isinstance(image, torch.Tensor):
        # Handle torch tensors: (3, H, W) or (H, W, 3)
        if image.ndim == 3:
            if image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                image = image.permute(1, 2, 0)
            # Convert to numpy and ensure [0, 255] range
            image_np = image.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        else:
            raise ValueError(f"Expected 3D tensor, got shape {image.shape}")
    elif Image is not None and isinstance(image, Image.Image):
        # Convert PIL Image to numpy array
        image_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        # Already numpy array, ensure correct format
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) array, got shape {image.shape}")
        image_np = image
        # Ensure uint8 type
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected PIL.Image, numpy.ndarray, or torch.Tensor"
        )

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1,  # 1 for full-range model (better for various distances)
        min_detection_confidence=min_detection_confidence,
    ) as face_detection:
        # Detect faces
        results = face_detection.process(image_np)

        # Check number of detected faces
        if results.detections is None or len(results.detections) == 0:
            raise ValueError(
                "No face detected in the image. "
                "Please provide an image with a clearly visible face."
            )

        if len(results.detections) > 1:
            raise ValueError(
                f"Multiple faces detected ({len(results.detections)} faces). "
                "Please provide an image with exactly one face."
            )

        # Extract bounding box from the single detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # MediaPipe returns: xmin, ymin, width, height (all normalized)
        x_min = bbox.xmin
        y_min = bbox.ymin
        x_max = bbox.xmin + bbox.width
        y_max = bbox.ymin + bbox.height

        # Clamp to [0, 1] range (in case of slight overshoots)
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        return (x_min, y_min, x_max, y_max)
