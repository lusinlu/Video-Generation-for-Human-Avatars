import os
import random
from pathlib import Path
from typing import Optional, Union
import yaml
import imageio
import argparse
import json
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as st_load_file
from PIL import Image
import torchvision.transforms.functional as TVF
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field

# from torch.serialization import add_safe_globals
# class TrainConfig:  # type: ignore
#     pass
# add_safe_globals([TrainConfig])


from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
import ltx_video.pipelines.crf_compressor as crf_compressor


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
        just_crop: If True, only crop the image to the target size without resizing
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    if not just_crop:
        image = image.resize((target_width, target_height))

    frame_tensor = TVF.to_tensor(image)  # PIL -> tensor (C, H, W), [0,1]
    frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
    frame_tensor_hwc = frame_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    frame_tensor_hwc = crf_compressor.compress(frame_tensor_hwc)
    frame_tensor = frame_tensor_hwc.permute(2, 0, 1) * 255.0  # (H, W, C) -> (C, H, W)
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def estimate_num_frames_from_text(text: str, frame_rate: int) -> int:
    # Simple heuristic: ~150 words/minute ≈ 2.5 words/sec
    words = max(1, len(text.strip().split()))
    duration_sec = max(1.0, words / 2.5)
    frames_raw = int(duration_sec * frame_rate)
    # Round to (8k + 1)
    frames = ((max(frames_raw, 9) - 1) // 8 + 1) * 8 + 1
    return frames


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_ltx_video_pipeline(
    ckpt_path: str,
    precision: str,
    transformer_path: str,
    sampler: Optional[str] = None,
    device: Optional[str] = None,
) -> LTXVideoPipeline:
    ckpt_path = Path(ckpt_path)

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)

    merged_sd = st_load_file(transformer_path, device="cpu")
    # Read transformer config from metadata if present
    with safe_open(transformer_path, framework="pt") as f:
        meta = f.metadata() or {}
        cfg_json = meta.get("config")
    cfg_obj = json.loads(cfg_json)["transformer"]
    # Build Transformer3DModel from config
    transformer = Transformer3DModel.from_config(cfg_obj)
    transformer.load_state_dict(merged_sd, strict=True)
    if precision == "bfloat16":
        transformer = transformer.to(torch.bfloat16)

    # Use constructor if sampler is specified, otherwise use from_pretrained
    if sampler == "from_checkpoint" or not sampler:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
    else:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )

    patchifier = SymmetricPatchifier(patch_size=1)
    # Move core modules to the selected device for compute
    transformer = transformer.to(device)
    vae = vae.to(device)
    vae = vae.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "scheduler": scheduler,
        "vae": vae,
        "allowed_inference_steps": allowed_inference_steps,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)

    return pipeline


def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)
    latent_upsampler.to(device)
    latent_upsampler.eval()
    return latent_upsampler


def load_pipeline_config(pipeline_config: str):
    current_file = Path(__file__)

    if os.path.isfile(current_file.parent / pipeline_config):
        path = current_file.parent / pipeline_config
    elif os.path.isfile(pipeline_config):
        path = pipeline_config
    else:
        raise ValueError(f"Pipeline config file {pipeline_config} does not exist")

    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class InferenceConfig:
    prompt: str = field(metadata={"help": "Prompt for the generation"})

    pipeline_config: str = field(metadata={"help": "Path to the pipeline config file"})
    input_media_path: Optional[str] = field(
        metadata={
            "help": "Path to the input image to be modified using the video-to-video pipeline"
        },
    )
    output_path: str = field(metadata={"help": "Path to save outputs"})
    transformer_path: str = field(metadata={"help": "Path to transformer weights"})


def infer(config: InferenceConfig):
    pipeline_config = load_pipeline_config(config.pipeline_config)

    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]
    if not os.path.isfile(ltxv_model_name_or_path):
        ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=ltxv_model_name_or_path,
            repo_type="model",
        )
    else:
        ltxv_model_path = ltxv_model_name_or_path

    spatial_upscaler_model_name_or_path = pipeline_config.get(
        "spatial_upscaler_model_path"
    )
    if spatial_upscaler_model_name_or_path and not os.path.isfile(
        spatial_upscaler_model_name_or_path
    ):
        spatial_upscaler_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=spatial_upscaler_model_name_or_path,
            repo_type="model",
        )
    else:
        spatial_upscaler_model_path = spatial_upscaler_model_name_or_path

    # Helper to get required config values from YAML
    def cfg_get_required(key):
        if key in pipeline_config:
            return pipeline_config[key]
        raise KeyError(f"Missing required key '{key}' in pipeline config")

    seed = cfg_get_required("seed")
    height = cfg_get_required("height")
    width = cfg_get_required("width")
    frame_rate = cfg_get_required("frame_rate")

    seed_everething(seed)

    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    decided_num_frames = estimate_num_frames_from_text(config.prompt or "", frame_rate)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((decided_num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    print(f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")

    device = get_device()

    precision = pipeline_config["precision"]
    sampler = pipeline_config.get("sampler", None)

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        sampler=sampler,
        device=device,
        transformer_path=config.transformer_path,
    )

    print(
        f"[inference] VAE decoder timestep_conditioning={getattr(pipeline.vae.decoder, 'timestep_conditioning', None)}"
    )

    if pipeline_config.get("pipeline_type", None) == "multi-scale":
        if not spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

    media_item = None
    if config.input_media_path:
        media_item = load_media_file(
            media_path=config.input_media_path,
            height=height,
            width=width,
            max_frames=num_frames_padded,
            padding=padding,
        )

    skip_layer_strategy = None

    # Prepare input for the pipeline
    sample = {
        "prompt": config.prompt,
    }

    generator = torch.Generator(device=device).manual_seed(seed)

    # Remove sizing/runtime keys from YAML to avoid duplicate kwargs
    filtered_cfg = {
        k: v
        for k, v in pipeline_config.items()
        if k not in ["height", "width", "num_frames"]
    }

    # Engage decoder last-step denoising by passing a small decode_timestep
    # (paper suggests decoder performs final step in pixel space)
    images = pipeline(
        **filtered_cfg,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        **sample,
        media_items=media_item,
        is_video=True,
        vae_per_channel_normalize=True,
        mixed_precision=(precision == "mixed_precision"),
        device=device,
    ).images

    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :decided_num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = frame_rate
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=seed,
                resolution=(height, width, decided_num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=seed,
                resolution=(height, width, decided_num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

        print(f"Output saved to {output_filename}")


def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:

    media_tensor = load_image_to_tensor_with_resize_and_crop(
        media_path, height, width, just_crop=just_crop
    )
    media_tensor = torch.nn.functional.pad(media_tensor, padding)
    return media_tensor


def cli_main():
    parser = argparse.ArgumentParser(description="LTX-Video avatar inference")
    parser.add_argument(
        "--pipeline_config",
        type=str,
        required=True,
        help="Path to inference yaml (e.g., configs/inference-avatars.yaml)",
    )
    parser.add_argument(
        "--input_media_path", type=str, required=True, help="Path to input image/video"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text (will be converted to audio latents)",
    )
    parser.add_argument(
        "--output_path", type=str, default="./result", help="Output directory"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="../outputs/avatars_checkpoints/model_merged_best.safetensors",
    )

    args = parser.parse_args()

    out_dir = args.output_path
    cfg = InferenceConfig(
        prompt=args.prompt,
        output_path=out_dir,
        pipeline_config=args.pipeline_config,
        input_media_path=args.input_media_path,
        transformer_path=args.transformer_path,
    )

    infer(cfg)


if __name__ == "__main__":
    cli_main()
