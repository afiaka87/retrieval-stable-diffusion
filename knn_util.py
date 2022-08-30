from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from clip_retrieval.clip_back import load_index
from diffusers import AutoencoderKL

VAE_MODEL_WEIGHTS = (
    "/home/claym/Projects/cog-stable-diffusion/diffusers-cache/vae"  # TODO
)


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
