import json
import pytest
import safetensors.torch
import torch

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
    create_video_autoencoder_demo_config,
    PER_CHANNEL_STATISTICS_PREFIX,
)
from ltx_video.models.transformers.transformer3d import Transformer3DModel


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, str):
        return f"{argname}-{val}"
    return f"{argname}-{repr(val)}"


@pytest.fixture
def num_latent_channels():
    return 16


@pytest.fixture
def video_autoencoder(num_latent_channels):
    config = create_video_autoencoder_demo_config(latent_channels=num_latent_channels)
    model = CausalVideoAutoencoder.from_config(config)
    model.eval().to(torch.bfloat16)
    return model


@pytest.fixture
def transformer_config(num_latent_channels):
    transformer_config = {
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 12,
        "attention_type": "default",
        "caption_channels": 4096,
        "cross_attention_dim": 192,
        "double_self_attention": False,
        "dropout": 0.0,
        "in_channels": num_latent_channels,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "num_attention_heads": 16,
        "num_embeds_ada_norm": 1000,
        "num_layers": 2,
        "num_vector_embeds": None,
        "only_cross_attention": False,
        "out_channels": num_latent_channels,
        "upcast_attention": False,
        "use_linear_projection": False,
        "qk_norm": "rms_norm",
        "standardization_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [120, 1, 1],
        "timestep_scale_multiplier": 1000,
    }
    return transformer_config


@pytest.fixture
def synthetic_ckpt_path(
    tmp_path, video_autoencoder, num_latent_channels, transformer_config
):
    # Create transformer
    transformer = Transformer3DModel.from_config(transformer_config)
    transformer.to(torch.bfloat16)

    # Prepare configs and state dicts
    configs = {"transformer": transformer_config, "vae": vars(video_autoencoder.config)}
    transformer_sd = transformer.state_dict()
    transformer_sd = {
        "model.diffusion_model." + key: value for key, value in transformer_sd.items()
    }

    # Prepare VAE state dict with per-channel statistics
    vae_sd = video_autoencoder.state_dict()
    vae_sd[f"{PER_CHANNEL_STATISTICS_PREFIX}std-of-means"] = torch.rand(
        num_latent_channels,
    )
    vae_sd[f"{PER_CHANNEL_STATISTICS_PREFIX}mean-of-means"] = torch.rand(
        num_latent_channels,
    )
    vae_sd = {"vae." + key: value for key, value in vae_sd.items()}

    out_file_path = f"{tmp_path}/test_ckpt.safetensors"
    safetensors.torch.save_file(
        {**transformer_sd, **vae_sd},
        out_file_path,
        metadata={"config": json.dumps(configs)},
    )
    return out_file_path
