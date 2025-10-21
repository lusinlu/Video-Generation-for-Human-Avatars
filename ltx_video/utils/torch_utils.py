import torch
from torch import nn
from safetensors.torch import save_file
import json
import copy
import os


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


def save_module_safetensors(module: nn.Module, target_path: str, metadata: dict | None = None) -> None:
    """Save an nn.Module's state_dict as a single safetensors file (CPU tensors).

    This does not mutate the module. All tensors are moved to CPU before writing.
    """
    state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    save_file(state, target_path, metadata=metadata or {})


def export_merged_safetensors(peft_model: nn.Module, target_path: str, metadata: dict | None = None) -> None:
    """Deepcopy a PEFT-wrapped model, merge LoRA into base weights, and save as safetensors.

    - Does NOT modify the original model (safe during training).
    - Requires the model to support `.merge_and_unload()` (PEFT).
    """
    model_copy = copy.deepcopy(peft_model)
    if not hasattr(model_copy, "merge_and_unload"):
        raise AttributeError("Model does not support merge_and_unload(); cannot export merged weights")
    merged = model_copy.merge_and_unload()
    # Build metadata with embedded config for single-file load
    meta = dict(metadata or {})
    try:
        cfg = getattr(merged, "config", None)
        if cfg is not None:
            # Extract plain dict of non-private fields
            cfg_dict = {k: v for k, v in getattr(cfg, "__dict__", {}).items() if not k.startswith("_")}
            # Merge optional scheduler config provided by caller
            scheduler_cfg = meta.pop("scheduler", None)
            config_root = {"transformer": cfg_dict}
            if scheduler_cfg is not None:
                config_root["scheduler"] = scheduler_cfg
            meta["config"] = json.dumps(config_root)
    except Exception:
        pass
    state = {k: v.detach().cpu() for k, v in merged.state_dict().items()}
    save_file(state, target_path, metadata=meta)


def save_training_checkpoint(
    model: nn.Module, 
    target_path: str, 
    train_mode: str,
    metadata: dict | None = None,
    is_best: bool = False
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
