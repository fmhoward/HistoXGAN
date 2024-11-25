"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace
import slideflow as sf

sys.path.append(".")
sys.path.append("..")

from e4e3.options.train_options import TrainOptions
from e4e3.training.coach_slideflow import Coach_Slideflow


def main():
    opts = TrainOptions().parse()
    previous_train_ckpt = None
    if opts.resume_training_from_ckpt:
        opts, previous_train_ckpt = load_train_checkpoint(opts)
    else:
        setup_progressive_steps(opts)
        create_initial_experiment_dir(opts)
    PROJECT_DIR = os.getcwd() + "/PROJECTS/HistoXGAN/"
    P = sf.Project(PROJECT_DIR)
    P.sources = [    "TCGA_ACC",
	  "TCGA_BLCA",
	    "TCGA_BRCA",
		  "TCGA_CESC",
		    "TCGA_CHOL",
			  "TCGA_COADREAD",
			    "TCGA_DLBC",
				  "TCGA_ESCA",
				    "TCGA_HNSC",
					  "TCGA_KICH",
					    "TCGA_KIRP",
                           "TCGA_KIRC",
						  "TCGA_LGG",
						    "TCGA_LIHC",
							  "TCGA_LUAD",
							    "TCGA_LUSC",
								  "TCGA_MESO",
								    "TCGA_OV",
									  "TCGA_PAAD",
									    "TCGA_PCPG",
										  "TCGA_PRAD",
										    "TCGA_SARC",
											  "TCGA_SKCM",
											    "TCGA_STAD",
												  "TCGA_TGCT",
												    "TCGA_THCA",
													  "TCGA_THYM",
													    "TCGA_UCEC",
														  "TCGA_UCS",
														    "TCGA_UVM"]
    P.annotations = PROJECT_DIR + "annotations_tcga_complete.csv"
    dataset = P.dataset(tile_px=512, tile_um=400)
    df_train = dataset.torch(batch_size=opts.batch_size, num_workers=int(opts.workers), drop_last = True)
    coach = Coach_Slideflow(opts, previous_train_ckpt, ds_train = df_train, ds_test = df_train)
    coach.train()


def load_train_checkpoint(opts):
    train_ckpt_path = opts.resume_training_from_ckpt
    previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
    new_opts_dict = vars(opts)
    opts = previous_train_ckpt['opts']
    opts['resume_training_from_ckpt'] = train_ckpt_path
    update_new_configs(opts, new_opts_dict)
    pprint.pprint(opts)
    opts = Namespace(**opts)
    if opts.sub_exp_dir is not None:
        sub_exp_dir = opts.sub_exp_dir
        opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
        create_initial_experiment_dir(opts)
    return opts, previous_train_ckpt


def setup_progressive_steps(opts):
    log_size = int(math.log(opts.stylegan_size, 2))
    num_style_layers = 2*log_size - 2
    num_deltas = num_style_layers - 1
    if opts.progressive_start is not None:  # If progressive delta training
        opts.progressive_steps = [0]
        next_progressive_step = opts.progressive_start
        for i in range(num_deltas):
            opts.progressive_steps.append(next_progressive_step)
            next_progressive_step += opts.progressive_step_every

    assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
        "Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
    return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
    if os.path.exists(opts.exp_dir):
        ind = 0
        base = opts.exp_dir
        while os.path.exists(opts.exp_dir):
            opts.exp_dir = base + str(ind)
            ind = ind + 1
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
    for k, v in new_opts.items():
        if k not in ckpt_opts:
            ckpt_opts[k] = v
    if new_opts['update_param_list']:
        for param in new_opts['update_param_list']:
            ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
    main()
