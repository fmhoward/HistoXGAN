# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from io import BytesIO
from typing import List, Optional
from tqdm import tqdm

import numpy as np
import PIL.Image
import slideflow as sf
import torch

from . import dnnlib, legacy

#----------------------------------------------------------------------------

class InvalidArgumentError(Exception):
    pass

#----------------------------------------------------------------------------

def generate_images(
    network_pkl: str,
    outdir: str,
    seeds: Optional[List[int]] = None,
    truncation_psi: float = 1.,
    noise_mode: str = 'const',
    format: str = 'png',
    class_idx: Optional[int] = None,
    projected_w: Optional[str] = None,
    save_projection: bool = False,
    resize: bool = False,
    gan_um: Optional[int] = None,
    gan_px: Optional[int] = None,
    target_um: Optional[int] = None,
    target_px: Optional[int] = None,
    slide_name: str = 'gan',
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    if format not in ('png', 'jpg'):
        raise InvalidArgumentError('--format must be either "jpg" or "png".')
    if resize:
        print("The `resize` argument is deprecated. To resize images, "
              "use the arguments `target_px` and `target_um`.")
    if target_px is not None and target_um is None:
        target_um = gan_um
    if target_px is not None:
        print(f"Resizing GAN images to target {target_px} px, {target_um} um")
    if target_px and None in (gan_um, gan_px, target_um, target_px):
        raise InvalidArgumentError('If resizing, must supply gan-um, gan-px, target-um, and target-px')

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # TFRecord writer.
    if sf.util.path_to_ext(outdir) == 'tfrecords':
        tfr_path = outdir
        outdir = os.path.dirname(outdir)
        print(f"Writing as TFRecords to {tfr_path}")
        writer = sf.io.TFRecordWriter(tfr_path)
    else:
        tfr_path = None

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            sf.log.warning('--seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        raise InvalidArgumentError('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise InvalidArgumentError('Must specify class label when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            sf.log.warning('--class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(tqdm(seeds)):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

        # Resize/crop image.
        if target_px:
            resize_factor = target_um / gan_um
            crop_width = int(resize_factor * gan_px)
            left = gan_px/2 - crop_width/2
            upper = gan_px/2 - crop_width/2
            right = left + crop_width
            lower = upper + crop_width
            image = image.crop((left, upper, right, lower)).resize((target_px, target_px))

        if tfr_path:
            slidename_bytes = bytes(slide_name, 'utf-8')
            with BytesIO() as output:
                image.save(output, format=format)
                record = sf.io.serialized_record(slidename_bytes, output.getvalue(), seed, 0)
            writer.write(record)

        else:
            image.save(f'{outdir}/seed{seed:04d}.{format}', quality=100)

        if save_projection:
            np.savez(f'{outdir}/projected_w_{seed:04d}.npz', w=z.cpu().numpy())

    if tfr_path:
        writer.close()
