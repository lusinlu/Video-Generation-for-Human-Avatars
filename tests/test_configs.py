import pytest
from pathlib import Path

from ltx_video.inference import infer, InferenceConfig

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


@pytest.fixture
def prompt():
    return "A video of a cat playing with a ball."


# mark as slow to avoid running these tests by default
@pytest.mark.slow
@pytest.mark.parametrize(
    "pipeline_config",
    [pytest.param(config, id=config.stem) for config in CONFIGS_DIR.glob("*.yaml")],
)
def test_run_config(tmp_path, prompt, pipeline_config):
    if "fp8" in pipeline_config.stem:
        pytest.skip("Skipping fp8 configs as they require specific hardware support.")

    inference_config = InferenceConfig(prompt=prompt)
    inference_config.pipeline_config = CONFIGS_DIR / pipeline_config
    inference_config.output_path = tmp_path / f"{pipeline_config.stem}"
    inference_config.height = 256
    inference_config.width = 320
    inference_config.num_frames = 33
    infer(config=inference_config)
