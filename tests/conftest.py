import pytest
import shutil
import os


@pytest.fixture(scope="session")
def test_paths(request, pytestconfig):
    try:
        output_path = "output"
        ckpt_path = request.param  # This will get the current parameterized item
        text_encoder_model_name_or_path = pytestconfig.getoption(
            "text_encoder_model_name_or_path"
        )
        input_image_path = pytestconfig.getoption("input_image_path")
        input_video_path = pytestconfig.getoption("input_video_path")

        config = {
            "ckpt_path": ckpt_path,
            "input_image_path": input_image_path,
            "input_video_path": input_video_path,
            "output_path": output_path,
            "text_encoder_model_name_or_path": text_encoder_model_name_or_path,
        }

        yield config

    finally:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)


def pytest_generate_tests(metafunc):
    if "test_paths" in metafunc.fixturenames:
        ckpt_paths = metafunc.config.getoption("ckpt_path")
        metafunc.parametrize("test_paths", ckpt_paths, indirect=True)


def pytest_addoption(parser):
    parser.addoption(
        "--ckpt_path",
        action="append",
        default=[],
        help="Path to checkpoint files (can specify multiple)",
    )
    parser.addoption(
        "--text_encoder_model_name_or_path",
        action="store",
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        help="Path to the checkpoint file",
    )
    parser.addoption(
        "--input_image_path",
        action="store",
        default="tests/utils/woman.jpeg",
        help="Path to input image file.",
    )
    parser.addoption(
        "--input_video_path",
        action="store",
        default="tests/utils/woman.mp4",
        help="Path to input video file.",
    )
