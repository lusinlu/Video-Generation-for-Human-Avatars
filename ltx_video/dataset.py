import torch
from pathlib import Path
from torch.utils.data import Dataset


def collate_latent_pairs(batch):
    """
    Custom collate function for latents, pose latents, and reference image latents.

    Args:
        batch: List of dicts with keys:
            - 'latents' [C, F, H, W] (main encoder latents)
            - 'pose_latents' [C, F, H, W] (pose frame latents)
            - 'ref_image_latents' [C, 1, H, W] (reference image latents)
            - 'stem' str

    Returns:
        Dict with:
            - latents: [B, C, F, H, W] (stacked)
            - pose_latents: [B, C, F, H, W] (stacked)
            - ref_image_latents: [B, C, 1, H, W] (stacked)
            - stem: List of strings
    """
    latents_list = [item["latents"] for item in batch]
    pose_latents_list = [item["pose_latents"] for item in batch]
    ref_image_latents_list = [item["ref_image_latents"] for item in batch]

    # Stack all latents
    latents_stacked = torch.stack(latents_list, dim=0)  # [B, C, F, H, W]
    pose_latents_stacked = torch.stack(pose_latents_list, dim=0)  # [B, C, F, H, W]
    ref_image_latents_stacked = torch.stack(
        ref_image_latents_list, dim=0
    )  # [B, C, 1, H, W]

    out = {
        "latents": latents_stacked,  # [B, C, F, H, W]
        "pose_latents": pose_latents_stacked,  # [B, C, F, H, W]
        "ref_image_latents": ref_image_latents_stacked,  # [B, C, 1, H, W]
        "stem": [item["stem"] for item in batch],
    }

    return out


class LatentPairDataset(Dataset):
    def __init__(
        self,
        condition_latents_dir: str,
        encoder_latents_dir: str,
    ):
        self.condition_dir = Path(condition_latents_dir)
        self.encoder_dir = Path(encoder_latents_dir)

        # Collect encoder latent files (*.pt) but exclude _ref.pt files
        encoder_files = sorted(self.encoder_dir.glob("*.pt"))
        encoder_files = [f for f in encoder_files if not f.stem.endswith("_ref")]

        self.items = []
        for ef in encoder_files:
            stem = ef.stem
            # Check that corresponding condition files exist
            pose_path = self.condition_dir / f"{stem}.pt"
            ref_path = self.condition_dir / f"{stem}_ref.pt"

            if pose_path.exists() and ref_path.exists():
                self.items.append(stem)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx]

        # Load main encoder latents from encoder_latents_dir
        encoder_path = self.encoder_dir / f"{stem}.pt"
        encoder_data = torch.load(encoder_path, map_location="cpu")
        latents = encoder_data["latents"].squeeze()  # [C, F, H, W]

        # Load pose latents from condition_latents_dir
        pose_path = self.condition_dir / f"{stem}.pt"
        pose_data = torch.load(pose_path, map_location="cpu")
        pose_latents = pose_data["latents"].squeeze()  # [C, F, H, W]

        # Load reference image latents from condition_latents_dir
        ref_path = self.condition_dir / f"{stem}_ref.pt"
        ref_data = torch.load(ref_path, map_location="cpu")
        ref_latents = ref_data["latents"].squeeze()  # [C, 1, H, W] or [C, H, W]
        # Ensure it has frame dimension
        if ref_latents.ndim == 3:
            ref_latents = ref_latents.unsqueeze(1)  # [C, 1, H, W]

        return {
            "latents": latents,  # [C, F, H, W] (main encoder latents)
            "pose_latents": pose_latents,  # [C, F, H, W] (pose frame latents)
            "ref_image_latents": ref_latents,  # [C, 1, H, W] (reference image latents)
            "stem": stem,
        }


class ValidationDataset(Dataset):
    def __init__(
        self,
        condition_latents_dir: str,
        encoder_latents_dir: str,
    ):
        self.condition_dir = Path(condition_latents_dir)
        self.encoder_dir = Path(encoder_latents_dir)

        encoder_files = sorted(self.encoder_dir.glob("*.pt"))
        encoder_files = [f for f in encoder_files if not f.stem.endswith("_ref")]

        self.items = []
        for ef in encoder_files:
            stem = ef.stem
            # Check that corresponding condition files exist
            pose_path = self.condition_dir / f"{stem}.pt"
            ref_path = self.condition_dir / f"{stem}_ref.pt"

            if pose_path.exists() and ref_path.exists():
                self.items.append(stem)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx]

        # Load main encoder latents from encoder_latents_dir
        encoder_path = self.encoder_dir / f"{stem}.pt"
        encoder_data = torch.load(encoder_path, map_location="cpu")
        latents = encoder_data["latents"].squeeze()  # [C, F, H, W]

        # Load pose latents from condition_latents_dir
        pose_path = self.condition_dir / f"{stem}.pt"
        pose_data = torch.load(pose_path, map_location="cpu")
        pose_latents = pose_data["latents"].squeeze()  # [C, F, H, W]

        # Load reference image latents from condition_latents_dir
        ref_path = self.condition_dir / f"{stem}_ref.pt"
        ref_data = torch.load(ref_path, map_location="cpu")
        ref_latents = ref_data["latents"].squeeze()  # [C, 1, H, W] or [C, H, W]
        # Ensure it has frame dimension
        if ref_latents.ndim == 3:
            ref_latents = ref_latents.unsqueeze(1)  # [C, 1, H, W]

        return {
            "latents": latents,  # [C, F, H, W] (main encoder latents)
            "pose_latents": pose_latents,  # [C, F, H, W] (pose frame latents)
            "ref_image_latents": ref_latents,  # [C, 1, H, W] (reference image latents)
            "stem": stem,
        }
