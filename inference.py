import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTX-Video")


def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return 0


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
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
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
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


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


def main():
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    # Directories
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        help="Path to the input video file (first frame used)",
    )
    parser.add_argument(
        "--input_image_path", type=str, help="Path to the input image file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the folder to save output video, if None will save in outputs/ directory.",
    )
    parser.add_argument("--seed", type=int, default="171198")

    # Pipeline parameters
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3,
        help="Guidance scale.",
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=1,
        help="Spatiotemporal guidance scale. 0 to disable STG.",
    )
    parser.add_argument(
        "--stg_rescale",
        type=float,
        default=0.7,
        help="Spatiotemporal guidance rescaling scale. 1 to disable rescale.",
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default="attention",
        help="Spatiotemporal guidance mode. "
        "It can be one of 'attention' (default), 'residual', or 'transformer_block'.",
    )
    parser.add_argument(
        "--stg_skip_layers",
        type=str,
        default="19",
        help="Layers to block for spatiotemporal guidance. Comma separated list of integers.",
    )
    parser.add_argument(
        "--image_cond_noise_scale",
        type=float,
        default=0.15,
        help="Amount of noise to add to the conditioned image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video frames. Optional if an input image provided.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Width of the output video frames. If None will infer from input image.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames to generate in the output video",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=25, help="Frame rate for the output video"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on. If not specified, will automatically detect and use CUDA or MPS if available, else CPU.",
    )
    parser.add_argument(
        "--precision",
        choices=["bfloat16", "mixed_precision"],
        default="bfloat16",
        help="Sets the precision for the transformer and tokenizer. Default is bfloat16. If 'mixed_precision' is enabled, it moves to mixed-precision.",
    )

    # VAE noise augmentation
    parser.add_argument(
        "--decode_timestep",
        type=float,
        default=0.05,
        help="Timestep for decoding noise",
    )
    parser.add_argument(
        "--decode_noise_scale",
        type=float,
        default=0.025,
        help="Noise level for decoding noise",
    )

    # Prompts
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt to guide generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt for undesired features",
    )

    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offloading unnecessary computations to CPU.",
    )

    parser.add_argument(
        "--text_encoder_model_name_or_path",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        help="Local path or model identifier for both the tokenizer and text encoder. Defaults to pretrained model on Hugging Face.",
    )

    # Add to the argument parser section
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["uniform", "linear-quadratic"],
        default=None,
        help="Sampler to use for noise scheduling. Can be either 'uniform' or 'linear-quadratic'. If not specified, uses the sampler from the checkpoint.",
    )

    args = parser.parse_args()
    logger.warning(f"Running generation with arguments: {args}")
    infer(**vars(args))


def create_ltx_video_pipeline(
    ckpt_path: str,
    precision: str,
    text_encoder_model_name_or_path: str,
    sampler: Optional[str] = None,
    device: Optional[str] = None,
) -> LTXVideoPipeline:
    ckpt_path = Path(ckpt_path)
    assert os.path.exists(
        ckpt_path
    ), f"Ckpt path provided (--ckpt_path) {ckpt_path} does not exist"
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    transformer = Transformer3DModel.from_pretrained(ckpt_path)

    # Use constructor if sampler is specified, otherwise use from_pretrained
    if sampler:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )
    else:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_model_name_or_path, subfolder="text_encoder"
    )
    patchifier = SymmetricPatchifier(patch_size=1)
    tokenizer = T5Tokenizer.from_pretrained(
        text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    vae = vae.to(torch.bfloat16)
    if precision == "bfloat16" and transformer.dtype != torch.bfloat16:
        transformer = transformer.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline


def infer(
    ckpt_path: str,
    output_path: Optional[str],
    seed: int,
    num_inference_steps: int,
    num_images_per_prompt: int,
    guidance_scale: float,
    stg_scale: float,
    stg_rescale: float,
    stg_mode: str,
    stg_skip_layers: str,
    image_cond_noise_scale: float,
    height: Optional[int],
    width: Optional[int],
    num_frames: int,
    frame_rate: int,
    precision: str,
    decode_timestep: float,
    decode_noise_scale: float,
    prompt: str,
    negative_prompt: str,
    offload_to_cpu: bool,
    text_encoder_model_name_or_path: str,
    input_image_path: Optional[str] = None,
    sampler: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
):
    device = device or get_device()

    if device == "mps":
        if torch.__version__ in ["2.4", "2.5"]:
            logger.warning(
                "PyTorch versions 2.4 and 2.5 have a known bug with MPS. Please update to PyTorch 2.6. More info: https://github.com/pytorch/pytorch/issues/141471"
            )

    generator = torch.Generator(device=device).manual_seed(seed)

    seed_everething(seed)
    if offload_to_cpu and not device == "cpu":
        logger.warning(
            "offload_to_cpu is set to True, but offloading will not occur since the model is already running on CPU."
        )
        offload_to_cpu = False
    else:
        offload_to_cpu = offload_to_cpu and get_total_gpu_memory() < 30

    output_dir = (
        Path(output_path)
        if output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=kwargs.get("device", get_device()),
    )

    conditioning_item = (
        prepare_conditioning(
            path=input_image_path,
            height=height,
            width=width,
            padding=padding,
        )
        if input_image_path
        else None
    )

    # Set spatiotemporal guidance
    skip_block_list = [int(x.strip()) for x in stg_skip_layers.split(",")]
    if stg_mode.lower() == "stg_a" or stg_mode.lower() == "attention":
        skip_layer_strategy = SkipLayerStrategy.Attention
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    skip_block_list = [int(x.strip()) for x in stg_skip_layers.split(",")]

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": conditioning_item,
    }

    images = pipeline(
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        skip_layer_strategy=skip_layer_strategy,
        skip_block_list=skip_block_list,
        stg_scale=stg_scale,
        do_rescaling=stg_rescale != 1,
        rescaling_scale=stg_rescale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if conditioning_item is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        image_cond_noise_scale=image_cond_noise_scale,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
        mixed_precision=(precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
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
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

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
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

        logger.warning(f"Output saved to {output_dir}")


def prepare_conditioning(
    path: str,
    height: int,
    width: int,
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    """Prepare conditioning items based on input media paths and their parameters.

    Args:
        path: Path to conditioning image
        strength: Strength for the conditioning media
        height: Height of the output frames
        width: Width of the output frames
        padding: Padding to apply to the frames

    Returns:
        A list of ConditioningItem objects.
    """

    frame_tensor = load_image_to_tensor_with_resize_and_crop(path, height, width)
    frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
    frame_tensor


if __name__ == "__main__":
    main()
