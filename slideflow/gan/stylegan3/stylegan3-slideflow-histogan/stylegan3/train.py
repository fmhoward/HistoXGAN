# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import copy
import click
import re
import json
import tempfile
import torch
from typing import Optional, Union, List

from . import dnnlib
from .training import training_loop
from .metrics import metric_main
from .torch_utils import training_stats
from .torch_utils import custom_ops

#----------------------------------------------------------------------------

def load_project(sf_kwargs):
    import slideflow as sf
    dataset_kwargs = {k:v for k,v in sf_kwargs.items() if k in ('tile_px', 'tile_um', 'filters', 'filter_blank', 'min_tiles')}
    project = sf.Project(sf_kwargs['project_path'])
    dataset = project.dataset(**dataset_kwargs)
    return project, dataset

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    if hasattr(c, 'slideflow_kwargs'):
        c_for_print = copy.deepcopy(c)
        c_for_print.training_set_kwargs.tfrecords = '[...]'
        c_for_print.training_set_kwargs.labels = '[...]'
    else:
        c_for_print = c

    print()
    print('Training options:')
    print(json.dumps(c_for_print, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    if hasattr(c, 'slideflow_kwargs'):
        print(f'Slideflow project:   {c.slideflow_kwargs.project_path}')
    else:
        print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Prepare slideflow dataset
    if hasattr(c, 'slideflow_kwargs'):
        print('Slideflow options:')
        print(json.dumps(c.slideflow_kwargs, indent=2))
        print('Setting up TFRecord indices...')
        project, dataset = load_project(c.slideflow_kwargs)
        dataset.build_index(False)

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        if not hasattr(c, 'slideflow_kwargs'):
            raise e
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


def init_slideflow_kwargs(path):
    try:
        with open(path, 'r') as sf_args_f:
            slideflow_kwargs = dnnlib.EasyDict(**json.load(sf_args_f))
    except IOError as err:
        raise click.ClickException(f'--slideflow: {err}')
    if slideflow_kwargs.model_type not in ('categorical', 'linear'):
        raise click.ClickException(f'Unknown slideflow model type {slideflow_kwargs.model_type}, must be "categorical" or "linear"')
    if slideflow_kwargs.model_type == 'linear':
        raise ValueError("Unsupported outcome type for conditional network: linear")
    project, dataset = load_project(slideflow_kwargs)
    outcome_key = 'outcomes' if 'outcomes' in slideflow_kwargs else 'outcome_label_headers'
    has_tile_labels = 'tile_labels' in slideflow_kwargs and slideflow_kwargs.tile_labels is not None
    if slideflow_kwargs[outcome_key] is not None:
        labels, unique = dataset.labels(slideflow_kwargs[outcome_key], use_float=(slideflow_kwargs['model_type'] != 'categorical'))
        if slideflow_kwargs.model_type == 'categorical':
            outcome_labels = dict(zip(range(len(unique)), unique))
        else:
            outcome_labels = None
    elif has_tile_labels:
        labels = None
        try:
            import pandas as pd
            _tl = pd.read_parquet(slideflow_kwargs.tile_labels)
            n_out = _tl.iloc[0].label.shape[0]
            out_range = list(map(str, range(n_out)))
            outcome_labels = dict(zip(out_range, out_range))
            del _tl
        except Exception as e:
            print(e)
            print("WARN: Unable to interpret tile labels for JSON logging.")
            raise
            outcome_labels = None
    else:
        labels = None
        outcome_labels = None

    # Configure the dataset interleaver
    if has_tile_labels:
        label_kwargs = dict(
            class_name='slideflow.io.torch.TileLabelInterleaver',
            tile_labels=slideflow_kwargs.tile_labels,
            labels=None,
        )
        slideflow_kwargs.outcome_label_headers = 'tile_labels'
    else:
        label_kwargs = dict(
            class_name='slideflow.io.torch.StyleGAN2Interleaver',
            labels=labels,
        )
    slideflow_kwargs.outcome_labels = outcome_labels

    # Normalizer
    if 'normalizer_kwargs' in slideflow_kwargs:
        label_kwargs.update(slideflow_kwargs.normalizer_kwargs)
        method = slideflow_kwargs.normalizer_kwargs['normalizer']
        print(f"Using {method} normalization.")

    if slideflow_kwargs.resize:
        final_size = slideflow_kwargs.resize
    elif slideflow_kwargs.crop:
        final_size = slideflow_kwargs.crop
    else:
        final_size = slideflow_kwargs.tile_px

    training_set_kwargs = dnnlib.EasyDict(
        tfrecords=dataset.tfrecords(),
        img_size=final_size,
        resolution=final_size,
        chunk_size=4,
        augment='xyr',
        standardize=False,
        num_tiles=dataset.num_tiles,
        max_size=dataset.num_tiles,  # Required for stylegan, not used by slideflow
        prob_weights=dataset.prob_weights,
        model_type=slideflow_kwargs.model_type,
        onehot=True,
        use_labels=(labels is not None or has_tile_labels),
        crop=slideflow_kwargs.crop,
        resize=slideflow_kwargs.resize,
        **label_kwargs
    )
    return training_set_kwargs, slideflow_kwargs, project.name

#----------------------------------------------------------------------------

def default_kwargs(
    outdir: str,
    cfg: str,
    data: Optional[str] = None,
    gpus: int = 1,
    batch: int = 32,
    gamma: float = 8.2,
    cond: bool = False,
    mirror: bool = False,
    aug: str = 'ada',
    resume: Optional[str] = None,
    freezed: int = 0,
    p: float = 0.2,
    target: float = 0.6,
    batch_gpu: Optional[int] = None,
    cbase: int = 32768,
    cmax: int = 512,
    glr: Optional[float] = None,
    dlr: Optional[float] = 0.002,
    map_depth: Optional[int] = None,
    mbstd_group: int = 4,
    desc: Optional[str] = None,
    metrics: Optional[Union[List, List[str]]] = None,
    kimg: int = 25000,
    tick: int = 4,
    snap: int = 50,
    seed: int = 0,
    fp32: bool = False,
    nobench: bool = False,
    workers: int = 3,
    dry_run: bool = False,
    slideflow: Optional[str] = None,
    lazy_resume: bool = False,
    train_histogan = False,
    feature_extractor = 'ctranspath',
    histo_lambda = 100
):
    return dict(locals())


def train(ctx=None, **kwargs):
    # Initialize config.
    kwargs = default_kwargs(**kwargs)
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    if opts.slideflow:
        c.training_set_kwargs, c.slideflow_kwargs, dataset_name = init_slideflow_kwargs(path=opts.slideflow)
    else:
        if opts.data is None:
            click.ClickException('--data required when not using a Slideflow dataset')
        c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics if opts.metrics is not None else []
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = opts.seed
    if not opts.slideflow:
        c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # HistoGAN training arguments
    c.loss_kwargs.train_histogan = opts.train_histogan
    c.loss_kwargs.feature_extractor = opts.feature_extractor
    c.loss_kwargs.histo_lambda = opts.histo_lambda
    
    # Lazy resume.
    c.lazy_resume = bool(opts.lazy_resume)

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
