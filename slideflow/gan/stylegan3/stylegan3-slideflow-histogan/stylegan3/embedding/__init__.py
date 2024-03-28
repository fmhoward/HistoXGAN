"""Embedding GAN functions."""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d

from .. import dnnlib, legacy
from ..training.networks_stylegan2 import EmbeddingGenerator, EmbeddingMappingNetwork


def load_embedding_gan(
    gan_pkl: str,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    print('Loading networks from "%s"...' % gan_pkl)
    with dnnlib.util.open_url(gan_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        if device is not None:
            G = G.to(device) # type: ignore
    G.mapping = EmbeddingMappingNetwork(G.mapping)
    return EmbeddingGenerator(G), G


def get_embeddings(
    G: torch.nn.Module,
    device: Optional[torch.device] = None
):
    embeddings = {}
    for e in range(G.c_dim):
        label = torch.zeros([1, G.c_dim], device=device)
        label[:, e] = 1
        embedding = G.mapping.embed(label).cpu().numpy()
        torch_embedding = torch.from_numpy(embedding)
        if device is not None:
            torch_embedding = torch_embedding.to(device)
        embeddings[e] = torch_embedding
    return embeddings


def load_gan_and_embeddings(
    gan_pkl: str,
    start: int,
    end: int,
    device: torch.device
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Load a GAN network and create Tensor embeddings.

    Args:
        gan_pkl (str): Path to GAN network pkl.
        start (int): Starting class index.
        end (int): Ending class index.
        device (torch.device): Device.

    Returns:
        torch.nn.Module: Embedding-interpolatable GAN module.

        torch.Tensor: First class embedding.

        torch.Tensor: Second class embedding.
    """
    E_G, G = load_embedding_gan(gan_pkl, device=device)
    embeddings = get_embeddings(G, device=device)
    return E_G, embeddings[start], embeddings[end]


def class_interpolate(
    E_G: torch.nn.Module,
    z: torch.tensor,
    embed0: Union[np.ndarray, torch.tensor],
    embed1: Union[np.ndarray, torch.tensor],
    device: torch.device,
    steps: int = 100,
    **gan_kwargs: Any
):
    if not isinstance(embed0, np.ndarray):
        embed0 = embed0.cpu().numpy()
        embed1 = embed1.cpu().numpy()
    interpolated_embedding = interp1d([0, steps-1], np.vstack([embed0, embed1]), axis=0)

    # Generate interpolated images.
    for interp_idx in range(steps):
        embed = torch.from_numpy(np.expand_dims(interpolated_embedding(interp_idx), axis=0)).to(device)
        img = E_G(z, embed, **gan_kwargs)
        img = (img + 1) * (255/2)
        img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        yield img


def linear_interpolate(
    G: torch.nn.Module,
    z: torch.tensor,
    device: torch.device,
    steps: int = 100,
    **gan_kwargs: Any
):
    for interp_idx in range(steps):
        torch_interp = torch.tensor([[interp_idx/steps]]).to(device)
        img = G(z, torch_interp, **gan_kwargs)
        img = (img + 1) * (255/2)
        img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        yield img
