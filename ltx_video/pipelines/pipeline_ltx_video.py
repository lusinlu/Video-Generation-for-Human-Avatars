# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
import inspect
import math
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from TTS.api import TTS

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import (
    un_normalize_latents,
    normalize_latents,
)


def _synthesize_audio(text: str, tts_model: str, out_wav_16k: Path) -> None:
    tts = TTS(model_name=tts_model)
    wav = tts.tts(text)
    src_sr = getattr(tts, "speakers_sample_rate", None) or 22050
    y = np.asarray(wav, dtype=np.float32)
    y16 = librosa.resample(y, orig_sr=int(src_sr), target_sr=16000)
    sf.write(str(out_wav_16k), y16, 16000, subtype="PCM_16")


def _faceformer_latents_from_text(
    text: str, device: str, tts_model: str = "tts_models/en/ljspeech/vits"
) -> Tuple[torch.Tensor, torch.Tensor]:
    from preprocessing.FaceFormer.faceformer import Faceformer  # local module

    tmp_dir = Path("outputs/tmp_audio")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / "tts.wav"
    _synthesize_audio(text, tts_model, wav_path)
    ff = Faceformer(device=device)
    ckpt_path = Path("../preprocessing/FaceFormer/vocaset.pth")
    sd = torch.load(str(ckpt_path), map_location=device)
    ff.load_state_dict(sd, strict=False)
    ff = ff.to(device).eval()
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000
        )
    input_values = np.asarray(audio, dtype=np.float32)
    if input_values.ndim == 1:
        input_values = np.reshape(input_values, (1, input_values.shape[0]))
    x = torch.FloatTensor(input_values).to(device)
    with torch.no_grad():
        lat = ff.extract_audio_motion_features(x)
    if torch.is_tensor(lat):
        lat = lat.detach().cpu().numpy()
    lat_t = torch.tensor(lat, dtype=torch.float32, device=device)
    if lat_t.ndim == 2:
        lat_t = lat_t.unsqueeze(0)
    attn_mask = torch.ones(lat_t.shape[:2], dtype=torch.long, device=device)
    # Aggressive offload: move FaceFormer back to CPU and clear CUDA cache if needed
    ff = ff.cpu()
    return lat_t, attn_mask


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    skip_initial_inference_steps: int = 0,
    skip_final_inference_steps: int = 0,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
            timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
            must be `None`.
        max_timestep ('float', *optional*, defaults to 1.0):
            The initial noising level for image-to-image/video-to-video. The list if timestamps will be
            truncated to start with a timestamp greater or equal to this.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

        if (
            skip_initial_inference_steps < 0
            or skip_final_inference_steps < 0
            or skip_initial_inference_steps + skip_final_inference_steps
            >= num_inference_steps
        ):
            raise ValueError(
                "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
            )

        timesteps = timesteps[
            skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
        ]
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        num_inference_steps = len(timesteps)

    return timesteps, num_inference_steps


class LTXVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using LTX-Video.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        transformer: Transformer3DModel,
        scheduler: DPMSolverMultistepScheduler,
        patchifier: Patchifier,
        allowed_inference_steps: Optional[List[float]] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
        )

        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.allowed_inference_steps = allowed_inference_steps
        print(
            f"[pipeline] VAE decoder timestep_conditioning={getattr(self.vae.decoder, 'timestep_conditioning', None)}"
        )

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        tts_model: str = "tts_models/en/ljspeech/vits",
        **kwargs,
    ):
        """
        Minimal override: instead of text-token embeddings, use FaceFormer audio latents from `prompt` text.
        Returns: (embeds, mask)
        """
        if device is None:
            device = self._execution_device
        assert isinstance(prompt, (str, list))
        prompt_embeds, prompt_attention_mask = _faceformer_latents_from_text(
            str(prompt), str(device), tts_model
        )

        # Match training-time audio latent sequence length by pad/truncate
        # Default to 96 if not provided
        max_len = int(kwargs.get("audio_max_steps", 96))
        T = int(prompt_embeds.shape[1])
        if T >= max_len:
            prompt_embeds = prompt_embeds[:, :max_len, :]
            prompt_attention_mask = torch.ones(
                (prompt_embeds.shape[0], max_len),
                dtype=torch.long,
                device=prompt_embeds.device,
            )
        else:
            pad = max_len - T
            pad_emb = torch.zeros(
                (prompt_embeds.shape[0], pad, prompt_embeds.shape[2]),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
            prompt_embeds = torch.cat([prompt_embeds, pad_emb], dim=1)
            pad_mask = torch.zeros(
                (prompt_attention_mask.shape[0], pad),
                dtype=torch.long,
                device=prompt_embeds.device,
            )
            prompt_attention_mask = torch.cat([prompt_attention_mask, pad_mask], dim=1)

        # Repeat for num_images_per_prompt
        bs, seq_len, dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
            bs * num_images_per_prompt, seq_len, dim
        )
        prompt_attention_mask = prompt_attention_mask.repeat(
            1, num_images_per_prompt
        ).view(bs * num_images_per_prompt, -1)
        return (prompt_embeds, prompt_attention_mask)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if prompt is None:
            raise ValueError(f"Cannot forward without `prompt`: {prompt}")

    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        latents: Optional[torch.Tensor],
        media_items: Optional[torch.Tensor],
        timestep: float,
        latent_shape: Union[torch.Size, Tuple[Any, ...]],
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
        vae_per_channel_normalize: bool = True,
    ):
        """
        Prepare the initial latent tensor to be denoised.
        The latents are either pure noise or a noised version of the encoded media items.
        Args:
            latents (`torch.FloatTensor` or `None`):
                The latents to use (provided by the user) or `None` to create new latents.
            timestep (`float`):
                The timestep to noise the encoded media_items to.
            latent_shape (`torch.Size`):
                The target latent shape.
            dtype (`torch.dtype`):
                The target dtype.
            device (`torch.device`):
                The target device.
            generator (`torch.Generator` or `List[torch.Generator]`):
                Generator(s) to be used for the noising process.
            vae_per_channel_normalize ('bool'):
                When encoding the media_items, whether to normalize the latents per-channel.
        Returns:
            `torch.FloatTensor`: The latents to be used for the denoising process. This is a tensor of shape
            (batch_size, num_channels, height, width).
        """
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {latent_shape[0]}. Make sure the batch size matches the length of the generators."
            )

        # Initialize the latents with the given latents or encoded media item, if provided
        assert (
            latents is None or media_items is None
        ), "Cannot provide both latents and media_items. Please provide only one of the two."

        # If media latents are provided with timestep >= 1.0, don't overwrite with pure noise.

        # Encode media items if provided
        if media_items is not None:
            assert isinstance(self.vae, CausalVideoAutoencoder)
            latents = vae_encode(
                media_items.to(dtype=self.vae.dtype, device=self.vae.device),
                self.vae,
                vae_per_channel_normalize=vae_per_channel_normalize,
            ).to(dtype=dtype, device=device)
            # If input is a single image (F=1), tile along time to match expected latent temporal length
            if latents is not None and latents.shape[2] != latent_shape[2]:
                if latents.shape[2] == 1 and latent_shape[2] > 1:
                    latents = latents.repeat(1, 1, latent_shape[2], 1, 1)
                else:
                    raise ValueError(
                        f"Encoded media has {latents.shape[2]} frames, but {latent_shape[2]} are required."
                    )

        if latents is not None:
            assert (
                latents.shape == latent_shape
            ), f"Latents have to be of shape {latent_shape} but are {latents.shape}."
            latents = latents.to(device=device, dtype=dtype)

        # For backward compatibility, generate in the "patchified" shape and rearrange
        b, c, f, h, w = latent_shape
        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # scale the initial noise by the standard deviation required by the scheduler
        noise = noise * self.scheduler.init_noise_sigma

        if latents is None:
            latents = noise
        else:
            # For img2vid, apply slight initial noising to allow motion even at high timesteps
            if media_items is not None:
                eps = 0.99
                latents = eps * noise + (1 - eps) * latents
            else:
                latents = timestep * noise + (1 - timestep) * latents

        return latents

    @staticmethod
    def classify_height_width_bin(
        height: int, width: int, ratios: dict
    ) -> Tuple[int, int]:
        """Returns binned height and width."""
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(
        samples: torch.Tensor, new_width: int, new_height: int
    ) -> torch.Tensor:
        n_frames, orig_height, orig_width = samples.shape[-3:]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Resize
            samples = LTXVideoPipeline.resize_tensor(
                samples, resized_height, resized_width
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[..., start_y:end_y, start_x:end_x]

        return samples

    @staticmethod
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)
        return media_items

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 20,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 4.5,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        stg_scale: Union[float, List[float]] = 1.0,
        rescaling_scale: Union[float, List[float]] = 0.7,
        guidance_timesteps: Optional[List[int]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        media_items: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        mixed_precision: bool = False,
        offload_to_cpu: bool = False,
        stochastic_sampling: bool = False,
        tone_map_compression_ratio: float = 0.0,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. If `timesteps` is provided, this parameter is ignored.
            skip_initial_inference_steps (`int`, *optional*, defaults to 0):
                The number of initial timesteps to skip. After calculating the timesteps, this number of timesteps will
                be removed from the beginning of the timesteps list. Meaning the highest-timesteps values will not run.
            skip_final_inference_steps (`int`, *optional*, defaults to 0):
                The number of final timesteps to skip. After calculating the timesteps, this number of timesteps will
                be removed from the end of the timesteps list. Meaning the lowest-timesteps values will not run.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_star_rescale (`bool`, *optional*, defaults to `False`):
                If set to `True`, applies the CFG star rescale. Scales the negative prediction according to dot
                product between positive and negative.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. This negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            enhance_prompt (`bool`, *optional*, defaults to `False`):
                If set to `True`, the prompt is enhanced using a LLM model.
            text_encoder_max_tokens (`int`, *optional*, defaults to `256`):
                The maximum number of tokens to use for the text encoder.
            stochastic_sampling (`bool`, *optional*, defaults to `False`):
                If set to `True`, the sampling is stochastic. If set to `False`, the sampling is deterministic.

            tone_map_compression_ratio: compression ratio for tone mapping, defaults to 0.0.
                        If set to 0.0, no tone mapping is applied. If set to 1.0 - full compression is applied.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        is_video = kwargs.get("is_video", False)
        self.check_inputs(prompt, height, width)

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        self.video_scale_factor = self.video_scale_factor if is_video else 1
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", True)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1
        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Prepare the list of denoising time-steps

        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = latent_shape

        assert skip_initial_inference_steps == 0 or latents is not None, (
            f"skip_initial_inference_steps ({skip_initial_inference_steps}) is used for image-to-image/video-to-video - "
            "media_item or latents should be provided."
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            skip_initial_inference_steps=skip_initial_inference_steps,
            skip_final_inference_steps=skip_final_inference_steps,
            **retrieve_timesteps_kwargs,
        )

        if self.allowed_inference_steps is not None:
            for timestep in [round(x, 4) for x in timesteps.tolist()]:
                assert (
                    timestep in self.allowed_inference_steps
                ), f"Invalid inference timestep {timestep}. Allowed timesteps are {self.allowed_inference_steps}."

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        # Normalize skip_block_list to always be None or a list of lists matching timesteps
        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if len(skip_block_list) == 0 or not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        # Prompt enhancement disabled in minimal avatar flow
        # 3. Encode input prompt (FaceFormer latents)
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            audio_max_steps=256,
        )

        self.transformer = self.transformer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        # 4. Prepare the initial latents using the provided media and conditioning items
        self.vae = self.vae.to(device)
        # Prepare the initial latents tensor, shape = (b, c, f, h, w)
        latents = self.prepare_latents(
            latents=latents,
            media_items=media_items,
            timestep=timesteps[0],
            latent_shape=latent_shape,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            vae_per_channel_normalize=vae_per_channel_normalize,
        )
        if offload_to_cpu and self.vae is not None:
            self.vae = self.vae.cpu()
        # No extra conditioning in minimal avatar flow: patchify latents and set defaults
        latents, latent_coords = self.patchifier.patchify(latents=latents)
        pixel_coords = latent_to_pixel_coords(
            latent_coords,
            self.vae,
            causal_fix=self.transformer.config.causal_temporal_positioning,
        )
        num_cond_latents = 0

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # Befor compiling this code please be aware:
        # This code might generate different input shapes if some timesteps have no STG or CFG.
        # This means that the codes might need to be compiled mutliple times.
        # To avoid that, use the same STG and CFG values for all timesteps.

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                do_classifier_free_guidance = guidance_scale[i] > 1.0
                do_spatio_temporal_guidance = stg_scale[i] > 0

                num_conds = 1
                if do_classifier_free_guidance:
                    num_conds += 1
                if do_spatio_temporal_guidance:
                    num_conds += 1

                indices = slice(0, batch_size)

                # Prepare skip layer masks
                skip_layer_mask: Optional[torch.Tensor] = None
                if do_spatio_temporal_guidance:
                    if skip_block_list is not None:
                        skip_layer_mask = self.transformer.create_skip_layer_mask(
                            batch_size, num_conds, num_conds - 1, skip_block_list[i]
                        )

                batch_pixel_coords = torch.cat([pixel_coords] * num_conds)
                fractional_coords = batch_pixel_coords.to(torch.float32)
                fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                # Choose the appropriate context manager based on `mixed_precision`
                if mixed_precision:
                    context_manager = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context_manager = nullcontext()  # Dummy context manager

                # predict noise model_output
                with context_manager:
                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=prompt_embeds_batch[indices].to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch[indices],
                        timestep=current_timestep,
                        skip_layer_mask=skip_layer_mask,
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # perform guidance
                # Minimal flow: no classifier-free guidance, no STG

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        if offload_to_cpu:
            self.transformer = self.transformer.cpu()
            if self._execution_device == "cuda":
                torch.cuda.empty_cache()

        # Remove the added conditioning latents
        latents = latents[:, num_cond_latents:]

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(self.patchifier.patch_size),
        )
        self.vae = self.vae.to(self._execution_device)
        # Debug/guard: ensure execution device and latents/vae align before decode

        print(
            "[dbg] exec_device=",
            self._execution_device,
            "vae_device=",
            self.vae.device,
            "latents_device=",
            latents.device,
        )

        if output_type != "latent":
            if self.vae.decoder.timestep_conditioning:
                noise = torch.randn_like(latents)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * latents.shape[0]
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * latents.shape[0]

                decode_timestep = torch.tensor(decode_timestep).to(latents.device)
                decode_noise_scale = torch.tensor(decode_noise_scale).to(
                    latents.device
                )[:, None, None, None, None]
                latents = (
                    latents * (1 - decode_noise_scale) + noise * decode_noise_scale
                )
            else:
                decode_timestep = None
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)
            # Ensure VAE is on same device as latents before decoding
            self.vae = self.vae.to(device)
            image = vae_decode(
                latents,
                self.vae,
                is_video,
                vae_per_channel_normalize=kwargs["vae_per_channel_normalize"],
                timestep=decode_timestep,
            )
            if offload_to_cpu and self.vae is not None:
                self.vae = self.vae.cpu()

            image = self.image_processor.postprocess(image, output_type=output_type)

        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:

        """
        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        return denoised_latents

    # removed: prepare_conditioning (not used in minimal avatar flow)

    @staticmethod
    def tone_map_latents(
        latents: torch.Tensor,
        compression: float,
    ) -> torch.Tensor:
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range
        in a perceptually smooth way using a sigmoid-based compression.

        This is useful for regularizing high-variance latents or for conditioning outputs
        during generation, especially when controlling dynamic behavior with a `compression` factor.

        Parameters:
        ----------
        latents : torch.Tensor
            Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
        compression : float
            Compression strength in the range [0, 1].
            - 0.0: No tone-mapping (identity transform)
            - 1.0: Full compression effect

        Returns:
        -------
        torch.Tensor
            The tone-mapped latent tensor of the same shape as input.
        """
        if not (0 <= compression <= 1):
            raise ValueError("Compression must be in the range [0, 1]")

        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered


def adain_filter_latent(
    latents: torch.Tensor, reference_latents: torch.Tensor, factor=1.0
):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
    statistics from a reference latent tensor.

    Args:
        latent (torch.Tensor): Input latents to normalize
        reference_latent (torch.Tensor): The reference latents providing style statistics.
        factor (float): Blending factor between original and transformed latent.
                       Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    result = latents.clone()

    for i in range(latents.size(0)):
        for c in range(latents.size(1)):
            r_sd, r_mean = torch.std_mean(
                reference_latents[i, c], dim=None
            )  # index by original dim order
            i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

            result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

    result = torch.lerp(latents, result, factor)
    return result


class LTXMultiScalePipeline:
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline, latent_upsampler: LatentUpsampler
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae
        self.latent_upsampler = latent_upsampler

    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict,
        second_pass: dict,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs.update(**first_pass)
        result = self.video_pipeline(*args, **kwargs)
        latents = result.images

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )

        kwargs = original_kwargs

        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = downscaled_width * 2
        kwargs["height"] = downscaled_height * 2
        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            result.images = videos

        return result
