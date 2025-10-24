from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class TrainConfig:
    checkpoint_path: str
    audio_latents_dir: Optional[str] = None
    encoder_latents_dir: Optional[str] = None
    # Optional validation sources
    val_audio_latents_dir: Optional[str] = None
    val_encoder_latents_dir: Optional[str] = None
    videos: Optional[str] = None

    output_dir: Optional[str] = None

    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None

    lora_rank: int = 8
    lora_alpha: int = 8

    precision: str = "bfloat16"

    # Audio encoder settings
    audio_embed_dim: int = 64  # Dimension of audio embeddings (FaceFormer output)

    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps

    use_deepspeed: bool = False  # Use DeepSpeed for model sharding
    deepspeed_config: Optional[str] = None  # Path to DeepSpeed config JSON
    local_rank: int = -1  # For distributed training (set by launcher)

    # RF scheduler params
    rf_num_train_timesteps: int = 1000
    rf_sampler: str = "Uniform"
    rf_shift: Optional[float] = None
    rf_shifting: Optional[str] = None
    rf_base_resolution: int = 32 * 32
    rf_target_shift_terminal: Optional[float] = None
    rf_log_normal_mu: Optional[float] = None
    rf_log_normal_sigma: Optional[float] = None
    # Clamp percentiles
    rf_quantile_min: float = 0.005
    rf_quantile_max: float = 0.999

    # Logging
    wandb_project: str = "ltx-video-avatars"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1

    # Decoder last-step training
    decoder_train: bool = False
    # Loss weights
    transformer_loss_weight: float = 1.0
    decoder_loss_l1_weight: float = 0.1
    decoder_loss_lpips_weight: float = 0.0
    decoder_t_max: float = 0.1


def load_train_config_from_yaml(yaml_path: str) -> TrainConfig:
    """
    Load training-related parameters from a YAML file.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg.get("checkpoint_path", None)
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required in YAML for training.")

    precision = cfg.get("precision", "bfloat16")

    sampler = cfg.get("sampler", None)
    rf_sampler = "Uniform"
    if isinstance(sampler, str):
        s = sampler.lower()
        if s == "uniform":
            rf_sampler = "Uniform"
        elif s in ("linear-quadratic", "linearquadratic"):
            rf_sampler = "LinearQuadratic"
        elif s == "from_checkpoint":
            rf_sampler = "Uniform"

    train_block = cfg.get("train", {}) or {}

    train_config = TrainConfig(
        checkpoint_path=checkpoint_path,
        precision=precision,
        audio_latents_dir=train_block.get("audio_latents_dir"),
        encoder_latents_dir=train_block.get("encoder_latents_dir"),
        val_audio_latents_dir=train_block.get("val_audio_latents_dir"),
        val_encoder_latents_dir=train_block.get("val_encoder_latents_dir"),
        videos=train_block.get("videos"),
        output_dir=train_block.get("output_dir"),
        batch_size=(
            int(train_block["batch_size"]) if "batch_size" in train_block else None
        ),
        num_epochs=(
            int(train_block["num_epochs"]) if "num_epochs" in train_block else None
        ),
        learning_rate=(
            float(train_block["learning_rate"])
            if "learning_rate" in train_block
            else None
        ),
        lora_rank=int(train_block.get("lora_rank", 8)),
        lora_alpha=int(train_block.get("lora_alpha", 8)),
        audio_embed_dim=int(train_block.get("audio_embed_dim", 64)),
        gradient_checkpointing=bool(train_block.get("gradient_checkpointing", False)),
        gradient_accumulation_steps=int(
            train_block.get("gradient_accumulation_steps", 1)
        ),
        use_deepspeed=bool(train_block.get("use_deepspeed", False)),
        deepspeed_config=train_block.get("deepspeed_config"),
        local_rank=int(train_block.get("local_rank", -1)),
        rf_sampler=rf_sampler,
        rf_num_train_timesteps=int(train_block.get("rf_num_train_timesteps", 1000)),
        rf_shift=(
            float(train_block.get("rf_shift")) if train_block.get("rf_shift") else None
        ),
        rf_shifting=train_block.get("rf_shifting"),
        rf_base_resolution=int(train_block.get("rf_base_resolution", 32 * 32)),
        rf_target_shift_terminal=(
            float(train_block.get("rf_target_shift_terminal"))
            if train_block.get("rf_target_shift_terminal")
            else None
        ),
        rf_log_normal_mu=(
            float(train_block.get("rf_log_normal_mu"))
            if train_block.get("rf_log_normal_mu")
            else None
        ),
        rf_log_normal_sigma=(
            float(train_block.get("rf_log_normal_sigma"))
            if train_block.get("rf_log_normal_sigma")
            else None
        ),
        rf_quantile_min=float(train_block.get("rf_quantile_min", 0.005)),
        rf_quantile_max=float(train_block.get("rf_quantile_max", 0.999)),
        wandb_project=train_block.get("wandb_project", "ltx-video-avatars"),
        wandb_run_name=train_block.get("wandb_run_name"),
        log_every_n_steps=int(train_block.get("log_every_n_steps", 10)),
        save_every_n_epochs=int(train_block.get("save_every_n_epochs", 1)),
        decoder_train=bool(train_block.get("decoder_train", False)),
        transformer_loss_weight=float(train_block.get("transformer_loss_weight", 1.0)),
        decoder_loss_l1_weight=float(train_block.get("decoder_loss_l1_weight", 0.1)),
        decoder_loss_lpips_weight=float(
            train_block.get("decoder_loss_lpips_weight", 0.0)
        ),
        decoder_t_max=float(train_block.get("decoder_t_max", 0.1)),
    )

    return train_config
