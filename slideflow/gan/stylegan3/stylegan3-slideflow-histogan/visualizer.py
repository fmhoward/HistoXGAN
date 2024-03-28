
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click

from stylegan3.visualizer import Visualizer

@click.command()
@click.argument('pkls', metavar='PATH', nargs=-1)
@click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
@click.option('--classifier', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--gan_um', help='Tile size in microns of GAN images', type=int)
@click.option('--gan_px', help='Tile size in pixels of GAN images', type=int)
@click.option('--target_um', help='Tile size in microns of target images', type=int)
@click.option('--target_px', help='Tile size in pixels of target images', type=int)
def main(
    pkls,
    capture_dir,
    browse_dir,
    classifier,
    gan_um,
    gan_px,
    target_um,
    target_px
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """

    if classifier is not None:
        if None in (gan_um, gan_px, target_um, target_px):
            raise click.ClickException("If supplying classifier model, must also supply "
                                       "gan_px, gan_um, target_px, target_um.")

        print("GAN image configuration:")
        print("Size (px):    ", gan_px)
        print("Size (um):    ", gan_um)

        viz = Visualizer(capture_dir=capture_dir, gan_px=gan_px, gan_um=gan_um)
        viz.load_model(classifier)
    else:
        viz = Visualizer(capture_dir=capture_dir)

    if browse_dir is not None:
        viz.pickle_widget.search_dirs = [browse_dir]

    # List pickles.
    if len(pkls) > 0:
        for pkl in pkls:
            viz.add_recent_pickle(pkl)
        viz.load_pickle(pkls[0])
    else:
        pretrained = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl'
        ]

        # Populate recent pickles list with pretrained model URLs.
        for url in pretrained:
            viz.add_recent_pickle(url)

    # Run.
    while not viz.should_close():
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
