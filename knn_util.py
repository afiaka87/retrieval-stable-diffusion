from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from clip_retrieval.clip_back import load_index
from diffusers import AutoencoderKL

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import functional as tvf
import clip


def clip_image_preprocess(image: torch.Tensor, clip_size: int = 224) -> torch.Tensor:
    image = tvf.resize(
        image, [clip_size, clip_size], interpolation=tvf.InterpolationMode.BICUBIC
    )
    image = tvf.to_tensor(image)
    image = (image + 1.0) / 2.0  # normalize to [0, 1]
    image = tvf.normalize(
        image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    )  # normalize to CLIP format
    return image


class Perceptor:
    def __init__(
        self,
        device,
        clip_name: str = "ViT-L/14",
    ):
        print(f"Loading CLIP model on {device}")
        self.device = device
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
        clip_model.eval()
        clip_model.to(device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_size = clip_model.visual.input_resolution

    @torch.no_grad()
    def encode_prompts(
        self, prompts: Union[str, List[str]], normalize: bool = True
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            # either a single prompt or a list of prompts separated by |
            prompts = prompts.split("|") if "|" in prompts else [prompts]
            prompts = [prompt.strip() for prompt in prompts]
        text_tokens = clip.tokenize(prompts).to(self.device)
        encoded_text = self.clip_model.encode_text(text_tokens)
        if normalize:
            encoded_text = encoded_text / torch.linalg.norm(
                encoded_text, dim=1, keepdim=True
            )
        if encoded_text.dim() == 2:
            encoded_text = encoded_text[:, None, :]
        return encoded_text

    @torch.no_grad()
    def encode_image(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        image = PILImage.open(image_path).convert("RGB")
        image = (
            clip_image_preprocess(image, clip_size=self.clip_size)
            .unsqueeze(0)
            .to(self.device)
        )
        image_features = self.clip_model.encode_image(image)
        if normalize:
            image_features /= image_features.norm(dim=1, keepdim=True)
        if image_features.ndim == 2:
            image_features = image_features[:, None, :]
        return image_features


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def load_vae(vae_model_weights: str, device: str = "cpu") -> AutoencoderKL:
    """
    Load the VAE model.
    """
    vae = AutoencoderKL.from_pretrained(
        vae_model_weights,
        local_files_only=True,  # TODO support diffusers api auth
    )
    vae.eval()
    vae.to(device)
    return vae


def load_retrieval_index(faiss_index_dir: str) -> faiss.Index:
    """
    Load the retrieval index.
    """
    # load the clip retrieval index
    index_path = Path(faiss_index_dir)
    retrieval_index = load_index(
        str(index_path / "image.index"), enable_faiss_memory_mapping=True
    )
    # load the vae
    vae_embeds = np.load(str(index_path / "vae_emb.npy"))

    return retrieval_index, vae_embeds


def knn_search(
    clip_retrieval_index: faiss.Index,
    query_embedding: torch.FloatTensor,
    num_result_ids: int,
):
    """
    compute the knn search
     - pass a text index to use text modality, image index to use image modality
    """
    print(f"query: {query_embedding.shape}")
    if query_embedding.ndim == 3:  # (b, 1, d)
        query_embedding = query_embedding.squeeze(
            1
        )  # need to reduce to (b, d) for faiss
    query_embeddings = query_embedding.cpu().detach().numpy().astype(np.float32)
    distances, indices, result_embeddings = clip_retrieval_index.search_and_reconstruct(
        query_embeddings, num_result_ids
    )
    results = indices[0]
    nb_results = np.where(results == -1)[0]

    if len(nb_results) > 0:
        nb_results = nb_results[0]
    else:
        nb_results = len(results)
    result_indices = results[:nb_results]
    result_distances = distances[0][:nb_results]
    result_embeddings = result_embeddings[0][:nb_results]
    result_embeddings = normalized(result_embeddings)
    result_embeddings = torch.from_numpy(result_embeddings)
    return result_distances, result_indices, result_embeddings


def map_ids_to_vae_embeddings(
    indices: List[int],
    distances: List[float],
    vae_embeddings: np.ndarray,
) -> dict:
    max_id = vae_embeddings.shape[0]
    matching_embeds = []
    for key, (dist, ind) in enumerate(zip(distances, indices)):
        if ind < max_id:
            match = vae_embeddings[ind]
            if match is not None:
                match = torch.from_numpy(match)
                matching_embeds.append(match)
    matching_embeds = torch.stack(matching_embeds)
    return matching_embeds
