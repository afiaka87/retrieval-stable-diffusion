import argparse
import datetime
import io
import os
import re
import urllib
from pathlib import Path

import torch
import wandb
from diffusers import PNDMScheduler
from PIL import Image
from tqdm import tqdm

from retrieval_stable_diffusion import RetrievalStableDiffusionPipeline
import knn_util


def download_image(url):
    urllib_request = urllib.request.Request(url)
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    pil_image = Image.open(img_stream)
    return pil_image


def slugify(text):
    return re.sub(r"[^\w\s-]", "", text).strip().lower().replace(" ", "-")[:150]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a prompt")
    parser.add_argument("--prompt", type=str, help="Input prompt", default="")
    parser.add_argument(
        "--width",
        type=int,
        help="Width of output image",
        choices=[128, 256, 512, 768, 1024],
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of output image",
        choices=[128, 256, 512, 768],
        default=512,
    )
    parser.add_argument(
        "--prompt_strength",
        type=float,
        help="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
        default=0.8,
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        help="Number of images to output",
        default=4,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of denoising steps",
        default=100,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Scale for classifier-free guidance",
        default=7.5,
    )
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="output"
    )
    parser.add_argument(
        "--model_cache",
        type=str,
        help="Model cache directory",
        default="diffusers-cache",
    )
    parser.add_argument(
        "--faiss_index_dir",
        type=str,
        help="Path to retrieval index",
    )
    parser.add_argument(
        "--project", type=str, help="Wandb project name", default="stable-diffusion"
    )
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")
    return parser.parse_args()


@torch.cuda.amp.autocast()
@torch.inference_mode()
def main():
    args = parse_args()
    prompts = open(args.prompt, "r").readlines()
    prompts = [prompt.strip() for prompt in prompts]
    prompts = [prompt for prompt in prompts if prompt]
    print(f"There are {len(prompts)} prompts in the file.")
    output_dir = Path(
        args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    generation_dir = output_dir / "images"
    generation_dir.mkdir(parents=True, exist_ok=True)
    caption_dir = output_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)

    """Run a single prediction on the model"""
    if args.seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    else:
        seed = args.seed
    print(f"Using seed: {seed}")

    wandb_run = wandb.init(project=args.project)
    wandb_run.config.update(args.__dict__)

    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )

    retrieval_index, retrieval_latents = knn_util.load_retrieval_index(
        args.faiss_index_dir
    )

    retrieval_stable_diffusion_pipeline = (
        RetrievalStableDiffusionPipeline.from_pretrained(
            args.model_cache,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=args.model_cache,
            local_files_only=True,
        ).to(args.device)
    )

    generator = torch.Generator(args.device).manual_seed(seed)

    with torch.autocast(device_type=args.device):
        for prompt_index, prompt in enumerate(prompts):
            output = retrieval_stable_diffusion_pipeline(
                prompt=prompt,
                retrieval_index=retrieval_index,
                retrieval_latents=retrieval_latents,
                width=args.width,
                height=args.height,
                prompt_strength=args.prompt_strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )
            output_paths = []
            for batch_index, generation in enumerate(output["sample"]):
                generation_stub = Path(f"{prompt_index:05d}_{batch_index:03d}")
                generation_path = generation_dir / generation_stub.with_suffix(".png")
                generation.save(generation_path)

                caption_path = caption_dir / generation_stub.with_suffix(".txt")
                caption_path.write_text(prompt)
                output_paths.append((generation_path, caption_path))
                wandb_run.log({"sample": wandb.Image(generation, caption=prompt)})
                tqdm.write(f"Saved {generation_path} - {prompt}")


if __name__ == "__main__":
    main()
