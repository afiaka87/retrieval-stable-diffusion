import inspect
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision.transforms import functional as torchvision_functional
from tqdm.auto import tqdm

import knn_util


from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor  # , CLIPTextModel, CLIPTokenizer


class RetrievalStableDiffusionPipeline(DiffusionPipeline):
    """
    Modified from https://github.com/huggingface/diffusers/pull/241
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.clip_perceptor = knn_util.Perceptor(
            device=self.device, clip_name="ViT-L/14"
        )
        print(f"Loaded CLIP ViT-L/14 on {self.device}")
        self.uncond_latents = self.clip_perceptor.encode_prompts([""]).to(
            self.device
        )  # for CFG

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        retrieval_index: faiss.Index,
        retrieval_latents: np.ndarray,
        width: int,
        height: int,
        prompt_strength: float = 0.8,
        clip_retrieved_weight: float = 0.4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Image:
        batch_size = 1  # TODO
        if prompt_strength < 0 or prompt_strength > 1:
            raise ValueError(
                f"The value of prompt_strength should in [0.0, 1.0] but is {prompt_strength}"
            )

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must both be divisible by 8")

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self.clip_perceptor.encode_prompts([prompt] * batch_size)

        # compute the knn search
        # results are sorted by knn similarity. the first n are the best scoring.
        # num_result_ids is set to 40, but may return less.
        distances, indices, clip_image_features = knn_util.knn_search(
            retrieval_index, text_embeddings[0], num_result_ids=batch_size
        )
        clip_image_features = clip_image_features.to(self.device)
        print(f"KNN search result: {indices}")

        # the knn_search returns integer indices,
        # we still need to look up the respective embeddings in our passed in `retrieval_latents`
        vae_latents = knn_util.map_ids_to_vae_embeddings(
            indices, distances, retrieval_latents
        )
        vae_latents = vae_latents.to(self.device)
        print(f"vae_latents: {vae_latents.shape}")

        # the starting diffusion timestep and the total number of timesteps
        # are related to the prompt_strength.
        t_start, timesteps = self.compute_timesteps(
            num_inference_steps, prompt_strength, offset, batch_size
        )
        timesteps = timesteps.to(self.device)

        # Now that we know where we start,
        # we can add the correct amount of gaussian noise to the vae_latents.
        latents = self.scale_and_noise_latents(
            init_latents=vae_latents,
            timesteps=timesteps,
            batch_size=batch_size,
            generator=generator,
        )

        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([text_embeddings, self.uncond_latents])
            print(
                f"Doing classifier free guidance by concatenating text_embeddings and uncond_latents: {text_embeddings.shape}"
            )

        text_embeddings = text_embeddings.to(self.device)
        clip_image_features = clip_image_features / torch.linalg.norm(clip_image_features, dim=-1, keepdim=True)
        text_embeddings = text_embeddings + clip_image_features * clip_retrieved_weight

        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        decoded_image_tensor = self.vae.decode(latents)
        decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)
        decoded_image_npy = decoded_image_tensor.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        # safety_checker_input = self.clip_perceptor.encode_image(decoded_image_tensor)
        # TODO this requires loading clip twice
        pil_images = self.numpy_to_pil(decoded_image_npy)
        # safety_cheker_input = self.feature_extractor(unsafe_pil_images, return_tensors="pt").to(self.device)
        # safe_images_npy, has_nsfw_concept = self.safety_checker(images=decoded_image_npy, clip_input=safety_cheker_input.pixel_values)
        # safe_images_pil = self.numpy_to_pil(safe_images_npy)
        return {"sample": pil_images }#, "nsfw_content_detected": has_nsfw_concept}

    def compute_timesteps(
        self, num_inference_steps, prompt_strength, offset, batch_size
    ):
        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * prompt_strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size, dtype=torch.long, device=self.device
        )
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        return t_start, timesteps

    def scale_and_noise_latents(
        self,
        init_latents: torch.FloatTensor,
        timesteps: torch.Tensor,
        batch_size: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        # encode the init image into latents and scale the latents
        init_latents = init_latents * 0.18215
        init_latents = torch.cat([init_latents] * batch_size)
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        noised_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
        return noised_latents
