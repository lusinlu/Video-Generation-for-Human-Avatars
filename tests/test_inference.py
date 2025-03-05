import pytest
import torch
from inference import infer, create_ltx_video_pipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, str):
        return f"{argname}-{val}"
    return f"{argname}-{repr(val)}"


@pytest.mark.parametrize(
    "conditioning_test_mode",
    ["unconditional", "first-frame", "first-sequence", "sequence-and-frame"],
    ids=lambda x: f"conditioning_test_mode={x}",
)
def test_infer_runs_on_real_path(test_paths, conditioning_test_mode):
    conditioning_params = {}
    if conditioning_test_mode == "unconditional":
        pass
    elif conditioning_test_mode == "first-frame":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_image_path"]
        ]
        conditioning_params["conditioning_start_frames"] = [0]
    elif conditioning_test_mode == "first-sequence":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_video_path"]
        ]
        conditioning_params["conditioning_start_frames"] = [0]
    elif conditioning_test_mode == "sequence-and-frame":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_video_path"],
            test_paths["input_image_path"],
        ]
        conditioning_params["conditioning_start_frames"] = [16, 32]
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_test_mode}")
    test_paths = {
        k: v
        for k, v in test_paths.items()
        if k not in ["input_image_path", "input_video_path"]
    }

    params = {
        "seed": 42,
        "num_inference_steps": 1,
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "stg_rescale": 0.7,
        "stg_mode": "attention_values",
        "stg_skip_layers": "1,2,3",
        "image_cond_noise_scale": 0.15,
        "height": 288,
        "width": 512,
        "num_frames": 33,
        "frame_rate": 25,
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "prompt": "A young woman with wavy, shoulder-length light brown hair stands outdoors on a foggy day. She wears a cozy pink turtleneck sweater, with a serene expression and piercing blue eyes. A wooden fence and a misty, grassy field fade into the background, evoking a calm and introspective mood.",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "offload_to_cpu": False,
        "prompt_enhancement_max_words": 1000,
        "prompt_enhancer_image_caption_model_name_or_path": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        "prompt_enhancer_llm_model_name_or_path": "unsloth/Llama-3.2-3B-Instruct",
    }

    infer(**{**conditioning_params, **test_paths, **params})

    # test without prompt enhancement
    params["prompt_enhancement_max_words"] = 0
    infer(**{**conditioning_params, **test_paths, **params})


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
        enhance_prompt=True,
        prompt_enhancer_image_caption_model_name_or_path="MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        prompt_enhancer_llm_model_name_or_path="unsloth/Llama-3.2-3B-Instruct",
    )

    params = {
        "seed": 42,
        "num_inference_steps": 2,
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "do_rescaling": True,
        "stg_scale": 1,
        "rescaling_scale": 0.7,
        "skip_layer_strategy": SkipLayerStrategy.AttentionValues,
        "skip_block_list": [1, 2],
        "image_cond_noise_scale": 0.15,
        "height": 288,
        "width": 512,
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


def test_prompt_enhancement(test_paths, monkeypatch):
    # Create pipeline with prompt enhancement enabled
    device = get_device()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=test_paths["ckpt_path"],
        device=device,
        precision="bfloat16",
        text_encoder_model_name_or_path=test_paths["text_encoder_model_name_or_path"],
        enhance_prompt=True,
        prompt_enhancer_image_caption_model_name_or_path="MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        prompt_enhancer_llm_model_name_or_path="unsloth/Llama-3.2-3B-Instruct",
    )

    original_prompt = "A cat sitting on a windowsill"

    # Mock the pipeline's _encode_prompt method to verify the prompt being used
    original_encode_prompt = pipeline.encode_prompt

    prompts_used = []

    def mock_encode_prompt(prompt, *args, **kwargs):
        prompts_used.append(prompt[0] if isinstance(prompt, list) else prompt)
        return original_encode_prompt(prompt, *args, **kwargs)

    pipeline.encode_prompt = mock_encode_prompt

    # Set up minimal parameters for a quick test
    params = {
        "seed": 42,
        "num_inference_steps": 1,
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "do_rescaling": True,
        "stg_scale": 1,
        "rescaling_scale": 0.7,
        "skip_layer_strategy": SkipLayerStrategy.AttentionValues,
        "skip_block_list": [1, 2],
        "image_cond_noise_scale": 0.15,
        "height": 288,
        "width": 512,
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

    # Run pipeline with prompt enhancement enabled
    _ = pipeline(
        prompt=original_prompt,
        negative_prompt="worst quality",
        enhance_prompt=True,
        **params,
    )

    # Verify that the enhanced prompt was used
    assert len(prompts_used) > 0
    assert (
        prompts_used[0] != original_prompt
    ), f"Expected enhanced prompt to be different from original prompt, but got: {original_prompt}"

    # Run pipeline with prompt enhancement disabled
    prompts_used.clear()
    _ = pipeline(
        prompt=original_prompt,
        negative_prompt="worst quality",
        enhance_prompt=False,
        **params,
    )

    # Verify that the original prompt was used
    assert len(prompts_used) > 0
    assert (
        prompts_used[0] == original_prompt
    ), f"Expected original prompt to be used, but got: {prompts_used[0]}"
