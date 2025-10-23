import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import imageio
import torch.nn.functional as F


def collate_latent_pairs(batch):
    """
    Custom collate function that pads audio latents to fixed length (256, like text encoder).
    
    Args:
        batch: List of dicts with keys 'audio_latents' [T, D] and 'latents' [C, F, H, W]
    
    Returns:
        Dict with:
            - audio_latents: [B, 256, D] (padded/truncated to fixed length)
            - audio_mask: [B, 256] (1 for real tokens, 0 for padding)
            - latents: [B, C, F, H, W] (stacked)
    """
    MAX_AUDIO_LENGTH = 256  # Match text encoder's max_length
    
    audio_latents_list = [item["audio_latents"] for item in batch]
    encoder_latents = [item["latents"] for item in batch]
    
    batch_size = len(batch)
    audio_dim = audio_latents_list[0].shape[-1]  # D dimension
    
    # Initialize padded tensors
    audio_latents_padded = torch.zeros(batch_size, MAX_AUDIO_LENGTH, audio_dim, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, MAX_AUDIO_LENGTH, dtype=torch.float32)
    
    for i, audio_lat in enumerate(audio_latents_list):
        seq_len = min(audio_lat.shape[0], MAX_AUDIO_LENGTH)
        # Copy actual data (truncate if longer than MAX_AUDIO_LENGTH)
        audio_latents_padded[i, :seq_len, :] = audio_lat[:seq_len, :]
        # Set mask to 1 for real tokens
        audio_mask[i, :seq_len] = 1.0
    
    # Stack encoder latents
    encoder_latents_stacked = torch.stack(encoder_latents, dim=0)
    
    out = {
        "audio_latents": audio_latents_padded,      # [B, 256, D]
        "audio_mask": audio_mask,                    # [B, 256]
        "latents": encoder_latents_stacked,          # [B, C, F, H, W]
    }
    if isinstance(batch[0], dict):
        if "target_video_path" in batch[0]:
            out["target_video_path"] = [item["target_video_path"] for item in batch]
        if "stem" in batch[0]:
            out["stem"] = [item["stem"] for item in batch]
    return out


class LatentPairDataset(Dataset):
    def __init__(self, audio_latents_dir: str, encoder_latents_dir: str, video_dir: str, video_ext: str = ".mp4"):
        self.audio_dir = Path(audio_latents_dir)
        self.encoder_dir = Path(encoder_latents_dir)
        self.video_dir = Path(video_dir)
        self.video_ext = video_ext
        # Collect encoder latent files (*.pt) and find matching audio+video files
        encoder_files = sorted(self.encoder_dir.glob("*.pt"))
        self.pairs = []
        for ef in encoder_files:
            stem = ef.stem
            audio_file = self.audio_dir / f"{stem}_ff.npy"
            video_file = self.video_dir / f"{stem}{self.video_ext}"
            # Strict: assume both exist (no try/catch, per request)
            if audio_file.exists() and video_file.exists():
                self.pairs.append((str(audio_file), str(ef), str(video_file)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        audio_path, encoder_path, video_path = rec[0], rec[1], rec[2]
        # Load FaceFormer latents (audio): shape [B, T, D] or [1, T, D] -> squeeze batch if needed
        audio_lat = np.load(audio_path)
        if audio_lat.ndim == 3 and audio_lat.shape[0] == 1:
            audio_lat = audio_lat[0]  # [T, D]
        audio_lat = torch.tensor(audio_lat, dtype=torch.float32)
        # Load VAE encoder latents: stored as {"latents": tensor [C, F, H, W]}
        encoder_data = torch.load(encoder_path, map_location="cpu")
        encoder_lat = encoder_data["latents"].squeeze()  # [C, F, H, W]
        # Load full target video as tensor [C, F, H, W], normalized to [0,1]
        reader = imageio.get_reader(video_path)
        frames = []
        for i in range(reader.count_frames()):
            frames.append(reader.get_data(i))
        reader.close()
        tgt_np = np.stack(frames, axis=0)  # [F, H, W, C]
        tgt = torch.from_numpy(tgt_np).permute(3, 0, 1, 2).float() / 255.0  # [C,F,H,W]

        return {
            "audio_latents": audio_lat,      # [T, D]
            "latents": encoder_lat,          # [C, F, H, W]
            "target_video": tgt,             # [C, F, H, W]
            "target_video_path": video_path, # str
            "stem": Path(encoder_path).stem,
        }


class ValidationDataset(Dataset):
    def __init__(self, audio_latents_dir: str, encoder_latents_dir: str, target_video_dir: str, video_ext: str = ".mp4"):
        self.audio_dir = Path(audio_latents_dir)
        self.encoder_dir = Path(encoder_latents_dir)
        self.video_dir = Path(target_video_dir)
        self.video_ext = video_ext

        encoder_files = sorted(self.encoder_dir.glob("*.pt"))
        self.pairs = []
        for ef in encoder_files:
            stem = ef.stem
            audio_file = self.audio_dir / f"{stem}_ff.npy"
            video_file = self.video_dir / f"{stem}{self.video_ext}"
            if audio_file.exists() and video_file.exists():
                self.pairs.append((str(audio_file), str(ef), str(video_file)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, encoder_path, video_path = self.pairs[idx]
        audio_lat = np.load(audio_path)
        if audio_lat.ndim == 3 and audio_lat.shape[0] == 1:
            audio_lat = audio_lat[0]
        audio_lat = torch.tensor(audio_lat, dtype=torch.float32)

        encoder_data = torch.load(encoder_path, map_location="cpu")
        encoder_lat = encoder_data["latents"].squeeze()

        return {
            "audio_latents": audio_lat,      # [T, D]
            "latents": encoder_lat,          # [C, F, H, W]
            "target_video_path": video_path, # str
            "stem": Path(encoder_path).stem,
        }

