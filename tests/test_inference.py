import pytest
import torch
from inference import infer, create_ltx_video_pipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, str):
        return f"{argname}-{val}"
    return f"{argname}-{repr(val)}"


@pytest.mark.parametrize(
    "do_txt_to_image", [True, False], ids=lambda x: f"do_txt_to_image={x}"
)
def test_infer_runs_on_real_path(test_paths, do_txt_to_image):
    # Update the input_image_path in test_paths if there is an override
    if do_txt_to_image:
        test_paths["input_image_path"] = None

    params = {
        "input_video_path": None,
        "seed": 42,
        "num_inference_steps": 1,
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "stg_rescale": 0.7,
        "stg_mode": "stg_a",
        "stg_skip_layers": "1,2,3",
        "image_cond_noise_scale": 0.15,
        "height": 480,
        "width": 704,
        "num_frames": 33,
        "frame_rate": 25,
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "prompt": "A vintage yellow car drives along a wet mountain road, its rear wheels kicking up a light spray as it moves. The camera follows close behind, capturing the curvature of the road as it winds through rocky cliffs and lush green hills. The sunlight pierces through scattered clouds, reflecting off the car's rain-speckled surface, creating a dynamic, cinematic moment. "
        "The scene conveys a sense of freedom and exploration as the car disappears into the distance.",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "offload_to_cpu": False,
    }
    infer(**{**test_paths, **params})


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everething(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def test_pipeline_on_batch(test_paths):
    device = get_device()

    pipeline = create_ltx_video_pipeline(
        ckpt_path=test_paths["ckpt_path"],
        device=device,
        precision="bfloat16",
        text_encoder_model_name_or_path=test_paths["text_encoder_model_name_or_path"],
    )

    params = {
        "input_video_path": None,
        "seed": 42,
        "num_inference_steps": 2,
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "do_rescaling": True,
        "stg_scale": 1,
        "rescaling_scale": 0.7,
        "skip_layer_strategy": SkipLayerStrategy.Attention,
        "skip_block_list": [1, 2],
        "image_cond_noise_scale": 0.15,
        "height": 480,
        "width": 704,
        "num_frames": 1,
        "frame_rate": 25,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "offload_to_cpu": False,
        "output_type": "pt",
        "is_video": False,
        "vae_per_channel_normalize": True,
        "mixed_precision": False,
    }

    first_prompt = "A vintage yellow car drives along a wet mountain road, its rear wheels kicking up a light spray as it moves. The camera follows close behind, capturing the curvature of the road as it winds through rocky cliffs and lush green hills. The sunlight pierces through scattered clouds, reflecting off the car's rain-speckled surface, creating a dynamic, cinematic moment. The scene conveys a sense of freedom and exploration as the car disappears into the distance."
    second_prompt = "A woman with blonde hair styled up, wearing a black dress with sequins and pearl earrings, looks down with a sad expression on her face. The camera remains stationary, focused on the woman's face. The lighting is dim, casting soft shadows on her face. The scene appears to be from a movie or TV show."

    sample = {
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_attention_mask": None,
        "negative_prompt_attention_mask": None,
        "media_items": None,
    }

    def get_images(prompts):
        generators = [
            torch.Generator(device=device).manual_seed(params["seed"]) for _ in range(2)
        ]
        seed_everething(params["seed"])

        images = pipeline(
            prompt=prompts,
            device=device,
            generator=generators,
            **sample,
            **params,
        ).images
        return images

    batch_diff_images = get_images([first_prompt, second_prompt])
    batch_same_images = get_images([second_prompt, second_prompt])

    # Take the second image from both runs
    image2_not_same = batch_diff_images[1, :, 0, :, :]
    image2_same = batch_same_images[1, :, 0, :, :]

    # Compute mean absolute difference, should be 0
    mad = torch.mean(torch.abs(image2_not_same - image2_same)).item()
    print(f"Mean absolute difference: {mad}")

    assert torch.allclose(image2_not_same, image2_same)
