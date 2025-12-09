import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union
import yaml
from transformers import HfArgumentParser
import imageio
import json
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
import torchvision.transforms.functional as TVF
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field

import ltx_video.pipelines.crf_compressor as crf_compressor
from ltx_video.generate_faceformer_frames import generate_faceformer_frames
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.utils.torch_utils import detect_face_bbox

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


def create_transformer(
    ckpt_path: str, precision: str, patchifier: SymmetricPatchifier = None
) -> Transformer3DModel:
    if precision == "bfloat16":
        return Transformer3DModel.from_pretrained(ckpt_path, patchifier=patchifier).to(
            torch.bfloat16
        )
    else:
        return Transformer3DModel.from_pretrained(ckpt_path, patchifier=patchifier)


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

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    patchifier = SymmetricPatchifier(patch_size=1)
    transformer = create_transformer(ckpt_path, precision, patchifier=patchifier)

    # Use constructor if sampler is specified, otherwise use from_pretrained
    if sampler == "from_checkpoint" or not sampler:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
    else:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )

    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_model_name_or_path, subfolder="text_encoder"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    vae = vae.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
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

    path = None
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
    text: str = field(metadata={"help": "Text to pronounce"})

    output_path: str = field(
        default_factory=lambda: Path(
            f"outputs/{datetime.today().strftime('%Y-%m-%d')}"
        ),
        metadata={"help": "Path to the folder to save the output video"},
    )

    # Pipeline settings
    pipeline_config: str = field(
        default="configs/ltxv-13b-0.9.7-dev.yaml",
        metadata={"help": "Path to the pipeline config file"},
    )
    seed: int = field(
        default=171198, metadata={"help": "Random seed for the inference"}
    )
    height: int = field(
        default=192, metadata={"help": "Height of the output video frames"}
    )
    width: int = field(
        default=320, metadata={"help": "Width of the output video frames"}
    )
    num_frames: int = field(
        default=121,
        metadata={"help": "Number of frames to generate in the output video"},
    )
    frame_rate: int = field(
        default=20, metadata={"help": "Frame rate for the output video"}
    )
    offload_to_cpu: bool = field(
        default=False, metadata={"help": "Offloading unnecessary computations to CPU."}
    )
    negative_prompt: str = field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        metadata={"help": "Negative prompt for undesired features"},
    )

    # Video-to-video arguments
    input_media_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the input video (or image) to be modified using the video-to-video pipeline"
        },
    )

    # Conditioning
    image_cond_noise_scale: float = field(
        default=0.0,
        metadata={"help": "Amount of noise to add to the conditioned image"},
    )
    conditioning_media_paths: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of paths to conditioning media (images or videos). Each path will be used as a conditioning item."
        },
    )
    conditioning_strengths: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "List of conditioning strengths (between 0 and 1) for each conditioning item. Must match the number of conditioning items."
        },
    )


def infer(config: InferenceConfig):
    pipeline_config = load_pipeline_config(config.pipeline_config)

    ltxv_model_path = pipeline_config["checkpoint_path"]

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

    conditioning_media_paths = config.conditioning_media_paths
    config.num_frames = len(os.listdir(conditioning_media_paths[1]))

    seed_everething(config.seed)
    if config.offload_to_cpu and not torch.cuda.is_available():
        logger.warning(
            "offload_to_cpu is set to True, but offloading will not occur since the model is already running on CPU."
        )
        offload_to_cpu = False
    else:
        offload_to_cpu = config.offload_to_cpu and get_total_gpu_memory() < 30

    output_dir = (
        Path(config.output_path)
        if config.output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((config.height - 1) // 32 + 1) * 32
    width_padded = ((config.width - 1) // 32 + 1) * 32
    # num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1
    num_frames_padded = config.num_frames

    padding = calculate_padding(
        config.height, config.width, height_padded, width_padded
    )

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    device = get_device()

    precision = pipeline_config["precision"]
    text_encoder_model_name_or_path = pipeline_config["text_encoder_model_name_or_path"]
    sampler = pipeline_config.get("sampler", None)

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=device,
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

    conditioning_items = (
        prepare_conditioning(
            conditioning_media_paths=conditioning_media_paths,
            height=config.height,
            width=config.width,
            padding=padding,
        )
        if conditioning_media_paths
        else None
    )

    stg_mode = pipeline_config.get("stg_mode", "attention_values")
    del pipeline_config["stg_mode"]
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    # Prepare input for the pipeline
    sample = {
        "prompt": config.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": config.negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    generator = torch.Generator(device=device).manual_seed(config.seed)

    images = pipeline(
        **pipeline_config,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=config.frame_rate,
        **sample,
        media_items=media_item,
        conditioning_items=conditioning_items,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=config.image_cond_noise_scale,
        mixed_precision=(precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
        device=device,
        ref_image=conditioning_items[0],
        pose_frames=conditioning_items[1],
    ).images

    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, : config.num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = config.frame_rate
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

        logger.warning(f"Output saved to {output_filename}")


def prepare_conditioning(
    conditioning_media_paths: List[str],
    height: int,
    width: int,
    padding: tuple[int, int, int, int],
) -> Optional[List]:
    """Prepare conditioning items based on input media paths and their parameters.

    Args:
        conditioning_media_paths: List of paths to conditioning media (images or videos)
        conditioning_strengths: List of conditioning strengths for each media item
        height: Height of the output frames
        width: Width of the output frames
        num_frames: Number of frames in the output video
        padding: Padding to apply to the frames
        pipeline: LTXVideoPipeline object used for condition video trimming

    Returns:
        A list of ConditioningItem objects.
    """
    conditioning_items = []
    for path in conditioning_media_paths:

        media_tensor = load_media_file(
            media_path=path,
            height=height,
            width=width,
            padding=padding,
            just_crop=False,
        )
        conditioning_items.append(media_tensor)
    return conditioning_items


def load_media_file(
    media_path: str,
    height: int,
    width: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:
    """Load media file from path (image, video, or folder of images).

    Args:
        media_path: Path to image file, video file, or folder containing images
        height: Target height
        width: Target width
        max_frames: Maximum number of frames to load
        padding: Padding to apply
        just_crop: Whether to just crop without resizing

    Returns:
        Tensor of shape (1, 3, num_frames, height, width)
    """
    # Check if it's a folder with images
    if os.path.isdir(media_path):
        # Load all image files from folder
        image_files = sorted(
            [
                f
                for f in os.listdir(media_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
        )

        if not image_files:
            raise ValueError(f"No image files found in folder: {media_path}")

        # Use all frames if max_frames is None, otherwise limit to max_frames
        num_input_frames = len(image_files)

        # Read and preprocess frames from folder
        frames = []
        for i in range(num_input_frames):
            frame_path = os.path.join(media_path, image_files[i])
            frame = Image.open(frame_path).convert("RGB")
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)

        # Stack frames along the temporal dimension
        media_tensor = torch.cat(frames, dim=2)

    else:
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        media_tensor = torch.nn.functional.pad(media_tensor, padding)

    return media_tensor


def main():
    parser = HfArgumentParser(InferenceConfig)
    config = parser.parse_args_into_dataclasses()[0]

    # Load image and detect face bbox
    conditioning_image = Image.open(config.conditioning_media_paths[0]).convert("RGB")
    face_bbox = detect_face_bbox(conditioning_image)

    # Generate FaceFormer frames with face positioned according to bbox
    motion_frames_path = generate_faceformer_frames(
        text=config.text,
        output_dir=os.path.join(config.output_path, "faceformer_frames"),
        face_bbox=face_bbox,
    )
    config.conditioning_media_paths.append(motion_frames_path)
    infer(config=config)


if __name__ == "__main__":
    main()
