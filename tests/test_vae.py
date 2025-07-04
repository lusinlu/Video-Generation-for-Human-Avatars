import pytest
import torch
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)


def test_encode_decode_shape(video_autoencoder, num_latent_channels):
    spatial_factor = video_autoencoder.spatial_downscale_factor
    temporal_factor = video_autoencoder.temporal_downscale_factor
    input_videos = torch.randn(2, 3, 17, 64, 64, dtype=torch.bfloat16)

    # Encode
    latent = video_autoencoder.encode(input_videos).latent_dist.mode()
    expected_shape = (
        input_videos.shape[0],
        num_latent_channels,
        (input_videos.shape[2] + 7) // temporal_factor,
        input_videos.shape[3] // spatial_factor,
        input_videos.shape[4] // spatial_factor,
    )
    assert latent.shape == expected_shape

    # Decode
    timestep = torch.ones(input_videos.shape[0]) * 0.1
    reconstructed_videos = video_autoencoder.decode(
        latent, target_shape=input_videos.shape, timestep=timestep
    ).sample
    assert input_videos.shape == reconstructed_videos.shape


def test_temporal_causality(video_autoencoder):
    # validate temporal causality in encoder
    input_videos = torch.randn(2, 3, 17, 64, 64, dtype=torch.bfloat16)
    latent = video_autoencoder.encode(input_videos).latent_dist.mode()

    # Check that encoding a single frame matches the corresponding slice in the full latent
    input_image = input_videos[:, :, :1, :, :]
    image_latent = video_autoencoder.encode(input_image).latent_dist.mode()
    assert torch.allclose(image_latent, latent[:, :, :1, :, :], atol=1e-6)

    # Check that encoding a sequence of frames matches the corresponding slice in the full latent
    input_sequence = input_videos[:, :, :9, :, :]
    sequence_latent = video_autoencoder.encode(input_sequence).latent_dist.mode()
    assert torch.allclose(sequence_latent, latent[:, :, :2, :, :], atol=1e-6)


@pytest.mark.parametrize(
    "layer_name,expected_temporal_factor,expected_spatial_factor",
    [
        ("compress_space_res", 1, 2),
        ("compress_space", 1, 2),
        ("compress_time_res", 2, 1),
        ("compress_time", 2, 1),
        ("compress_all_res", 2, 2),
        ("compress_all", 2, 2),
    ],
)
def test_downscale_factors(
    num_latent_channels, layer_name, expected_temporal_factor, expected_spatial_factor
):
    patch_size = 4
    encoder_blocks = [
        (layer_name, {"multiplier": 2}),
    ]
    decoder_blocks = [
        ("compress_all", {"residual": True, "multiplier": 2}),
    ]
    config = {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "encoder_blocks": encoder_blocks,
        "decoder_blocks": decoder_blocks,
        "latent_channels": num_latent_channels,
        "norm_layer": "pixel_norm",
        "patch_size": patch_size,
        "latent_log_var": "uniform",
        "use_quant_conv": False,
        "causal_decoder": False,
        "timestep_conditioning": True,
        "spatial_padding_mode": "replicate",
    }
    model = CausalVideoAutoencoder.from_config(config)
    assert model.temporal_downscale_factor == expected_temporal_factor
    assert model.spatial_downscale_factor == expected_spatial_factor * patch_size
