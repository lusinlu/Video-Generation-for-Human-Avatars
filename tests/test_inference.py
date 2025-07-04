from dataclasses import asdict
import pytest
import torch
import yaml

from ltx_video.inference import (
    create_ltx_video_pipeline,
    get_device,
    infer,
    InferenceConfig,
)
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


@pytest.fixture
def input_image_path():
    return "tests/utils/woman.jpeg"


@pytest.fixture
def input_video_path():
    return "tests/utils/woman.mp4"


def base_inference_config(tmp_path, pipeline_config):
    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(pipeline_config, f)

    return InferenceConfig(
        seed=42,
        height=256,
        width=320,
        num_frames=49,
        frame_rate=25,
        prompt="A young woman with wavy, shoulder-length light brown hair stands outdoors on a foggy day. She wears a cozy pink turtleneck sweater, with a serene expression and piercing blue eyes. A wooden fence and a misty, grassy field fade into the background, evoking a calm and introspective mood.",
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        output_path=tmp_path,
        pipeline_config=temp_config_path,
    )


@pytest.fixture
def base_pipeline_config(synthetic_ckpt_path):
    return {
        "num_inference_steps": 1,
        "stg_mode": "attention_values",
        "skip_block_list": [1],
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "checkpoint_path": synthetic_ckpt_path,
        "text_encoder_model_name_or_path": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "prompt_enhancer_image_caption_model_name_or_path": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        "prompt_enhancer_llm_model_name_or_path": "unsloth/Llama-3.2-3B-Instruct",
        "prompt_enhancement_words_threshold": 120,
        "sampler": "LinearQuadratic",
    }


@pytest.mark.parametrize(
    "conditioning_test_mode",
    ["unconditional", "first-frame", "first-sequence", "sequence-and-frame"],
    ids=lambda x: f"conditioning_test_mode={x}",
)
def test_condition_modes(
    tmp_path,
    conditioning_test_mode,
    input_image_path,
    input_video_path,
    base_pipeline_config,
):
    inference_config = base_inference_config(tmp_path, base_pipeline_config)

    if conditioning_test_mode == "unconditional":
        pass
    elif conditioning_test_mode == "first-frame":
        inference_config.conditioning_media_paths = [input_image_path]
        inference_config.conditioning_start_frames = [0]
    elif conditioning_test_mode == "first-sequence":
        inference_config.conditioning_media_paths = [input_video_path]
        inference_config.conditioning_start_frames = [0]
    elif conditioning_test_mode == "sequence-and-frame":
        inference_config.conditioning_media_paths = [input_video_path, input_image_path]
        inference_config.conditioning_start_frames = [16, 43]
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_test_mode}")

    # Test that the infer function runs without errors
    infer(inference_config)


def test_vid2vid(tmp_path, input_video_path, base_pipeline_config):
    pipeline_config = base_pipeline_config
    pipeline_config["num_inference_steps"] = 3
    pipeline_config["skip_initial_inference_steps"] = 1

    inference_config = base_inference_config(tmp_path, pipeline_config)
    inference_config.num_frames = 25
    inference_config.input_media_path = input_video_path

    # Test that the infer function runs without errors
    infer(inference_config)


def test_pipeline_on_batch(tmp_path, base_pipeline_config):
    pipeline_config = base_pipeline_config
    inference_config = base_inference_config(tmp_path, pipeline_config)
    inference_config.num_frames = 1  # For faster test, we use a single frame

    device = get_device()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=pipeline_config["checkpoint_path"],
        device=device,
        precision=pipeline_config["precision"],
        text_encoder_model_name_or_path=pipeline_config[
            "text_encoder_model_name_or_path"
        ],
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=pipeline_config[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        prompt_enhancer_llm_model_name_or_path=pipeline_config[
            "prompt_enhancer_llm_model_name_or_path"
        ],
        sampler="LinearQuadratic",
    )

    first_prompt = "A vintage yellow car drives along a wet mountain road, its rear wheels kicking up a light spray as it moves. The camera follows close behind, capturing the curvature of the road as it winds through rocky cliffs and lush green hills. The sunlight pierces through scattered clouds, reflecting off the car's rain-speckled surface, creating a dynamic, cinematic moment. The scene conveys a sense of freedom and exploration as the car disappears into the distance."
    second_prompt = "A woman with blonde hair styled up, wearing a black dress with sequins and pearl earrings, looks down with a sad expression on her face. The camera remains stationary, focused on the woman's face. The lighting is dim, casting soft shadows on her face. The scene appears to be from a movie or TV show."

    def get_images(prompts):
        generators = [
            torch.Generator(device=device).manual_seed(inference_config.seed)
            for _ in range(2)
        ]
        torch.manual_seed(inference_config.seed)

        params = asdict(inference_config)
        params["prompt"] = prompts

        pipeline_result = pipeline(
            generator=generators,
            output_type="pt",
            vae_per_channel_normalize=True,
            **params,
        )
        return pipeline_result.images

    # Run the pipeline on two different batches of prompts
    batch_diff_images = get_images([first_prompt, second_prompt])
    batch_same_images = get_images([second_prompt, second_prompt])

    # Take the second image from both runs, which should be equal
    image2_not_same = batch_diff_images[1, :, 0, :, :]
    image2_same = batch_same_images[1, :, 0, :, :]

    assert torch.allclose(image2_not_same, image2_same)


def test_prompt_enhancement(tmp_path, base_pipeline_config):
    pipeline_config = base_pipeline_config
    inference_config = base_inference_config(tmp_path, pipeline_config)
    inference_config.num_frames = 1  # For faster test, we use a single frame

    device = get_device()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=pipeline_config["checkpoint_path"],
        device=device,
        precision=pipeline_config["precision"],
        text_encoder_model_name_or_path=pipeline_config[
            "text_encoder_model_name_or_path"
        ],
        enhance_prompt=True,
        prompt_enhancer_image_caption_model_name_or_path=pipeline_config[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        prompt_enhancer_llm_model_name_or_path=pipeline_config[
            "prompt_enhancer_llm_model_name_or_path"
        ],
        sampler="LinearQuadratic",
    )
    # Mock the pipeline's _encode_prompt method to verify the prompt being used
    original_encode_prompt = pipeline.encode_prompt

    def mock_encode_prompt(prompt, *args, **kwargs):
        prompts_used.append(prompt[0] if isinstance(prompt, list) else prompt)
        return original_encode_prompt(prompt, *args, **kwargs)

    pipeline.encode_prompt = mock_encode_prompt

    original_prompt = "A cat sitting on a windowsill"
    inference_config.prompt = original_prompt

    def run_pipeline(enhance_prompt):
        params = asdict(inference_config)
        pipeline(
            enhance_prompt=enhance_prompt,
            **params,
            skip_layer_strategy=SkipLayerStrategy.AttentionValues,
            vae_per_channel_normalize=True,
            output_type="pt",
        )
        assert (
            len(prompts_used) > 0
        ), f"No prompts were used in the pipeline run with enhance_prompt={enhance_prompt}"

        if enhance_prompt:
            # Verify that the enhanced prompt was used
            assert (
                prompts_used[0] != original_prompt
            ), f"Expected enhanced prompt to be different from original prompt, but got: {original_prompt}"
        else:
            # Verify that the original prompt was used
            assert (
                prompts_used[0] == original_prompt
            ), f"Expected original prompt to be used, but got: {prompts_used[0]}"

    # Run pipeline with prompt enhancement enabled
    prompts_used = []
    run_pipeline(enhance_prompt=True)
    # Run pipeline with prompt enhancement disabled
    prompts_used = []
    run_pipeline(enhance_prompt=False)
