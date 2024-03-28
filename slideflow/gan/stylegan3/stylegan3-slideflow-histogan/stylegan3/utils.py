from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

import imageio
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer


def noise_tensor(seed: int, z_dim: int) -> torch.Tensor:
    """Creates a noise tensor based on a given seed and dimension size.

    Args:
        seed (int): Seed.
        z_dim (int): Dimension of noise vector to create.

    Returns:
        torch.Tensor: Noise vector of shape (1, z_dim)
    """
    return torch.from_numpy(np.random.RandomState(seed).randn(1, z_dim))


def masked_embedding(embedding_dims, embed_first, embed_second):
    mask = np.ones(embed_first.shape[1])
    for e in embedding_dims:
        mask[e] = 0
    inv_mask = (~mask.astype(bool)).astype(int)
    embed_second = embed_first * mask + embed_second * inv_mask
    return embed_second


def save_video(
    imgs: List[np.ndarray],
    path: str,
    fps: int = 30,
    codec:str = 'libx264',
    bitrate: str = '16M',
    macro_block_size: int = 1
) -> None:
    # Padd the image if the width/height is odd
    if imgs[0].shape[0] % 2:
        imgs = [np.pad(img, ((1, 0), (1, 0), (0, 0))) for img in imgs]
    video_file = imageio.get_writer(
        path,
        mode='I',
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        macro_block_size=macro_block_size
    )
    for img in imgs:
        video_file.append_data(img)
    video_file.close()


def save_merged(imgs: List[np.ndarray], path: str, steps: Optional[int] = None):
    if steps is None:
        steps = len(imgs)
    width = imgs[0].shape[1]
    out_img = Image.new('RGB', (width*steps, width))
    x_offset = 0
    for img in imgs:
        out_img.paste(Image.fromarray(img), (x_offset, 0))
        x_offset += width
    out_img.save(path)
