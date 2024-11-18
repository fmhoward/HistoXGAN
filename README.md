# HistoXGAN
HistoXGAN (<b>Histo</b>logy feature e<b>X</b>plainability <b>G</b>enerative <b>A</b>dversarial <b>N</b>etwork) is a tool to perform highly accurate reconstruction of histology from latent space features. This tool can be used to explain the predictions of deep learning models that rely on extracted image features from foundational encoders, or to translate genomics or radiographic images into histology to understand histologic reflections of genomics or perform a 'virtual tumor biopsy'.

<img src='https://github.com/fmhoward/HistoXGAN/blob/main/histoxgan_architecture.png?raw=true'>

## Attribution
If you use this code in your work or find it helpful, please consider citing our paper in <a href='https://www.science.org/doi/10.1126/sciadv.adq0856'>Science Advances</a>.
```
@article{howard_generative_2024,
	title = {Generative adversarial networks accurately reconstruct pan-cancer histology from pathologic, genomic, and radiographic latent features},
	volume = {10},
	url = {https://www.science.org/doi/10.1126/sciadv.adq0856},
	doi = {10.1126/sciadv.adq0856},
	abstract = {Artificial intelligence models have been increasingly used in the analysis of tumor histology to perform tasks ranging from routine classification to identification of molecular features. These approaches distill cancer histologic images into high-level features, which are used in predictions, but understanding the biologic meaning of such features remains challenging. We present and validate a custom generative adversarial network—HistoXGAN—capable of reconstructing representative histology using feature vectors produced by common feature extractors. We evaluate HistoXGAN across 29 cancer subtypes and demonstrate that reconstructed images retain information regarding tumor grade, histologic subtype, and gene expression patterns. We leverage HistoXGAN to illustrate the underlying histologic features for deep learning models for actionable mutations, identify model reliance on histologic batch effect in predictions, and demonstrate accurate reconstruction of tumor histology from radiographic imaging for a “virtual biopsy.”},
	number = {46},
	urldate = {2024-11-18},
	journal = {Science Advances},
	author = {Howard, Frederick M. and Hieromnimon, Hanna M. and Ramesh, Siddhi and Dolezal, James and Kochanny, Sara and Zhang, Qianchen and Feiger, Brad and Peterson, Joseph and Fan, Cheng and Perou, Charles M. and Vickery, Jasmine and Sullivan, Megan and Cole, Kimberly and Khramtsova, Galina and Pearson, Alexander T.},
	month = nov,
	year = {2024},
	note = {Publisher: American Association for the Advancement of Science},
	pages = {eadq0856},
```

## Installation
This github repository should be downloaded to a project directory. Installation takes < 5 minutes on a standard desktop computer. Runtime for HistoXGAN training across the entire TCGA dataset requires approximately 72 hours to run for 25,000 kimg on 4 A100 GPUs. Subsequent generation of images with the trained HistoXGAN network occurs in < 1 second per image. All software was tested on CentOS 8 with an AMD EPYC 7302 16-Core Processor and 4x A100 SXM 40 GB GPUs.

Requirements:
* python 3.8
* pytorch 1.11
* opencv 4.6.0.66
* scipy 1.9.3
* scikit-image 0.19.3
* OpenSlide
* Libvips 8.9
* pandas 1.5.1
* numpy 1.21.6

For full environment used for model testing please see the environment.yml file


## HistoXGAN Applications
### Visualization of Histology Feature Space
Visualizing synthetic histology from a feature vector is easily performed with HistoXGAN; the included models allow for visualization of CTransPath and RetCCL feature vectors.
Trained models used in this work are available at https://doi.org/10.5281/zenodo.10892176. The trained HistoXGAN models alone can be downloaded from the FINAL_MODELS.rar folder in this Zenodo repository; or the trained models in conjunction with other supplemental data used to evaluate HistoXGAN can be downloaded from the HistoXGAN.rar folder.


The following code illustrates reconstruction of an image from CTransPath feature vectors:
```
#Load the CTransPath HistoXGAN model
from slideflow.gan.stylegan3.stylegan3 import dnnlib, legacy, utils
#The link here should reference the HistoXGAN model location downloaded from Zenodo
with dnnlib.util.open_url('../FINAL_MODELS/CTransPath/snapshot.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

#Select a feature vector to visualize
#The /PROJECTS/HistoXGAN/SAVED_FEATURES folder contains extracted CTransPath feature vectors for visualization
df = pd.read_csv('../PROJECTS/HistoXGAN/SAVED_FEATURES/brca_features_part.csv')
feat_cols = list(df_mod.columns.values)
feat_cols = [f for f in feat_cols if 'Feature_' in f]

#Select the first feature vector in the dataset
device = torch.device('cuda:0')
vector_base = torch.tensor([df[feat_cols].loc[0, :].values.tolist()]).to(device)
img_gen = G(vector_base, 0, noise_mode ='const

#Convert image to 0 - 255 scale and proper W X H X C order for visualization
import matplotlib.pyplot as plt
img_show = ((img_gen + 1)*127.5).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
fig, ax = plt.subplots()
plt.imshow(img_show)
plt.show()
```

### Visualization of Transition from Low to High Grade
For a more complex example, we can identify the feature vector indicative of low / high grade and visualize transitions along that feature vector.

```
#The following code generates images along the transition for high / low grade

import pandas as pd
import numpy as np
import pickle
import os
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from slideflow.gan.stylegan3.stylegan3 import dnnlib, legacy, utils
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

PROJECT_DIR = os.getcwd()
device = torch.device('cuda:3')

with dnnlib.util.open_url(PROJECT_DIR + '/FINAL_MODELS/CTransPath/snapshot.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

def vector_interpolate(
    G: torch.nn.Module,
    z: torch.tensor,
    z2: torch.tensor,
    device: torch.device,
    steps: int = 100
):
    for interp_idx in range(steps):
        torch_interp = torch.tensor(z - z2 + 2*interp_idx/steps*z2).to(device)
        img = G(torch_interp, 0, noise_mode ='const')
        img = (img + 1) * (255/2)
        img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        yield img
        
def generate_images(vector_z, vector_z2, prefix = 'test'):
    z = torch.tensor(vector_z).to(device)
    z2 = torch.tensor(vector_z2).to(device)
    img_array = []
    generator = vector_interpolate(G, z, z2, device = device)
    for interp_idx, img in enumerate(generator):
        img_array += [img]
    return img_array

def get_log_features(df, col, name):
    y = df[[col]].values
    feat_cols = list(df.columns.values)
    feat_cols = [f for f in feat_cols if 'Feature_' in f]
    #print(feat_cols)
    X = df[feat_cols]
    vector_list = X.loc[0, :].values.tolist()        
    clf = LogisticRegression().fit(X, y)
    return vector_list, clf.coef_


def GRADE_DATASET(dataset, ind, grade_col):
    df = pd.read_csv(PROJECT_DIR + "/PROJECTS/HistoXGAN/SAVED_FEATURES/" +  dataset.lower() + "_features_slide.csv")
    df2 = pd.read_csv(PROJECT_DIR + "/PROJECTS/HistoXGAN/tcga_all_annotations.csv")
    df_mod = pd.read_csv(PROJECT_DIR + "/PROJECTS/HistoXGAN/SAVED_FEATURES/" +  dataset.lower() + "_features_part.csv")
    feat_cols = list(df.columns.values)
    feat_cols = [f for f in feat_cols if 'Feature_' in f]
    vector_base = df_mod[feat_cols].loc[ind, :].values.tolist()    
    df['patient'] = df['Slide'].str[0:12]
    df = df.merge(df2, left_on='patient', right_on='patient', how = 'left')
    df=df.dropna(subset=['high_grade'])
    df['Grade_Class'] = 0
    df.loc[df.high_grade == 'Y', 'Grade_Class'] = 1
    vector_z, vector_z2 = get_log_features(df, 'Grade_Class', 'Grade_' + dataset)
    vector_z = vector_base
    return generate_images(vector_z, vector_z2, prefix = 'Grade_' + dataset)

img_dict = {}
img_dict['BRCA'] = GRADE_DATASET('BRCA', 100, 'Grade')
img_dict['PAAD'] = GRADE_DATASET('PAAD', 100, 'histological_grade')
img_dict['HNSC'] = GRADE_DATASET('HNSC', 200, 'neoplasm_histologic_grade')
img_dict['PRAD'] = GRADE_DATASET('PRAD', 200, 'Clinical_Gleason_sum')

img_include = 7
fig, axs2 = plt.subplots(len(img_dict), img_include, figsize = (2*img_include, 2*len(img_dict)))

row_loc = {
    'BRCA':[20,30,40,50,60,70,80],
    'HNSC':[20,30,40,50,60,70,80],
    'PAAD':[20,30,40,50,60,70,80],
    'PRAD':[20,30,40,50,60,70,80],
}
img_include = 7
col = 0
for img_name in img_dict:
    row = 0
    for row_item in row_loc[img_name]:
        axs2[col][row].imshow(img_dict[img_name][row_item])
        axs2[col][row].set_xticks([])
        axs2[col][row].set_yticks([])
        axs2[col][row].xaxis.set_label_position('top')
        row = row + 1
    str_name = img_name
    axs2[col][0].set_ylabel(str_name, size = 18)
    col = col + 1
fig.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, wspace=0, hspace=0)
axs2[0][0].annotate(text="", xy=(1.00, 1.032), xytext=(0.535,1.032), xycoords="figure fraction",  arrowprops=dict(facecolor='C1'))
axs2[0][0].annotate(text="", xy=(0.05, 1.032), xytext=(0.525,1.032), xycoords="figure fraction",  arrowprops=dict(facecolor='C0'))
axs2[0][0].annotate(text="Grade", xy = (0.53,1.05), xycoords="figure fraction", ha="center", size = 18)
axs2[0][0].annotate(text="Low", xy = (0.05, 1.05), xycoords="figure fraction", ha="center", size = 16)
axs2[0][0].annotate(text="High", xy = (1.00, 1.05), xycoords="figure fraction", ha="center", size = 16)

plt.show()
```

Example output:
<img src='https://github.com/fmhoward/HistoXGAN/blob/main/grade_transition.png?raw=true'>


### More Complex Visualization Tasks
For reproducibility, we have provided code for all experiments conducted in our study are described in detail in the included Jupyter notebooks.
Figures (part 1) contains:
* Calculation of overall loss statistics for HistoXGAN and comparator models
* Comparison of model predictions on real / synthetic tiles for grade, subtype, and gene expression
* Illustration of traversal using gradient descent for models trained to predict PIK3CA and HRD status in TCGA-BRCA
* Illustration of traversal along individual PCA componenets for models trained to predict grade, ancestry, and tissue source site (illustrating batch effect)

Figures (part 2) contains:
* Traversal along attention / prediction axes from MIL models for grade and tumor subtype
* Regenerating histology from radiomic features extracted from MRI

## HistoXGAN and model training from scratch
### Setup
This package heavily utilizes the <a href='https://github.com/jamesdolezal/slideflow/tree/master/slideflow'>Slideflow repository</a>, and reading the associated <a href='https://slideflow.dev/'>extensive documentation<a> is recommended to familiarize users with the workflow used in this environment.

This code supports multiple methods of reproducing our work. For the most straightforward way, skip to the  'Replicating our Analysis' header, which provides detailed instructions on navigating the jupyter notebooks that performed statistical analysis of results. Otherwise, please continue reading for a more complete guide to replication.

After downloading the github repository, the first step is to set up the datasets.json (located in the main directory of this repository) to reflect the location of where slide images are stored for the TCGA, CPTAC, and UChicago datasets.

Each 'dataset' within the /PROJECTS/HistoXGAN/datasets.json has four elements which must be entered:
```
"slides": location of the whole slide images
"roi": location of region of interest annotations. We have provided our region of interest annotations used for this project in the /roi/ directory.
"tiles": location to extract free image tiles. We disable this in our extract image function
"tfrecords": location of tfrecords containing the extracted image tiles for slides
```
	
The TCGA slide images can be downloaded from <a href='https://portal.gdc.cancer.gov'>https://portal.gdc.cancer.gov</a>, and CPTAC slides can be downloaded from <a href='https://wiki.cancerimagingarchive.net/display/Public/CPTAC+Pathology+Slide+Downloads>TCIA</a>. The "roi" marker should point to the appropriate folder within the ROI directory - the ROI subfolder should be used for all TCGA models.

### Slide Extraction
Slide extraction from image slides should be performed as described in the <a href='https://slideflow.dev/slide_processing/'>slideflow documentation</a>. After setting up the datasets as above, slides can be extracted simply:
	
```	
import slideflow as sf
P = sf.Project('../PROJECTS/HistoXGAN')
P.annotations = '../PROJECTS/HistoXGAN/annotations_tcga_complete.csv'
P.sources = ['TCGA_ACC', 'TCGA_BLCA', ...]
P.extract_tiles(tile_px = 512, tile_um = 400)
```

To extract full slide images without regions of interest for MIL models used in the study:

```	
import slideflow as sf
P = sf.Project('../PROJECTS/HistoXGAN')
P.annotations = '../PROJECTS/HistoXGAN/annotations_tcga_complete.csv'
P.sources = ['TCGA_BLCA_NOROI', 'TCGA_BRCA_NOROI']
P.extract_tiles(tile_px = 512, tile_um = 400, roi_method = 'ignore')
```

### HistoXGAN training
Training of HistoXGAN is fully integrated into the slideflow repository. Of note, empirical testing has shown that the weight of the L1 loss lambda parameter (histo_lambda) should be adjusted so the magnitude of L1 loss is approximately 10 at the start of training (i.e. histo_lambda = 100 for CTransPath; 10 for UNI; etc). Higher values lead to diminished diversity of produced images; lower values lead to reduced learning of features.

```	
P = sf.Project('../PROJECTS/HistoXGAN')
P.annotations = '../PROJECTS/HistoXGAN/annotations_tcga_complete.csv'
P.sources = ['TCGA_ACC', 'TCGA_BLCA', ...]
dataset = P.dataset(tile_px=512, tile_um=400)
P.gan_train(
	dataset=dataset,
	model='stylegan3',
	cfg='stylegan2',
	exp_label="HistoXGAN_CTransPath",
	gpus=4,
	batch=32*8,
	batch_gpu=16,
	train_histogan = True,			#Indicator to train a HistoXGAN instead of a default StyleGAN
	feature_extractor = 'ctranspath',	#Specify any feature extractor implemented in slideflow
	histo_lambda = 100			#Specify weighting of the L1 loss between extracted features
)

```

### Alternative Encoder Training
We provide a modified verison of the encoder4editing to train the comparator encoders. This can be run as follows:
```
python train_e4e.py \
--exp_dir experiment/ctranspath \
--start_from_latent_avg \
--w_discriminator_lambda 0.0 \
--lambda_ctp 1.0 
--val_interval 10000 \
--max_steps 200000 \
--stylegan_size 512 \
--stylegan_weights path/to/pretrained/stylegan2.pt \
--workers 8 \
--batch_size 8 \
--test_batch_size 4 \
--test_workers 4
```
To train with a target of retccl feature vectors, use --lambda_retccl 1.0. To train an lpips/dists encoder, use --lambda_lpips 0.8 --lambda_dists 0.8. To train a single style encoder, use --encoder_type SingleStyleCodeEncoder

### Tile-based Model Training / Evaluation
All annotations used for model training (grade, subtype, single gene expression, ancestry) are in the '../PROJECTS/HistoXGAN/tcga_all_annotations.csv' file. For training a model on TCGA for three fold cross validation of the held out set:

```
P = sf.Project('../PROJECTS/HistoXGAN/')
P.sources = ["TCGA_BRCA"]
P.annotations = "../PROJECTS/HistoXGAN/tcga_all_annotations.csv"
P.train(outcomes="high_grade",
                params=sf.ModelParams(tile_px = 512, tile_um = 400, epochs=1,
                l2 = 1e-05,
                batch_size = 32,
                drop_images = False,
                dropout = 0.5,
                hidden_layer_width = 256,
                hidden_layers = 3,
                learning_rate = 0.0001,
                learning_rate_decay = 0.97,
                learning_rate_decay_steps = 100000,
                #loss = "sparse_categorical_crossentropy",
                model = "xception",
                optimizer = "Adam",
                pooling = "avg",
                toplayer_epochs = 0,
                trainable_layers = 0),
                filters={"high_grade": ["Y", "N"]},
                val_strategy='k-fold',
		val_k_fold = 3,
                mixed_precision = True)
```


For training a model on all of TCGA for validation on CPTAC:
```
P.train(outcomes="high_grade",
                params=sf.ModelParams(tile_px = 512, tile_um = 400, epochs=1,
                l2 = 1e-05,
                batch_size = 32,
                drop_images = False,
                dropout = 0.5,
                hidden_layer_width = 256,
                hidden_layers = 3,
                learning_rate = 0.0001,
                learning_rate_decay = 0.97,
                learning_rate_decay_steps = 100000,
                #loss = "sparse_categorical_crossentropy",
                model = "xception",
                optimizer = "Adam",
                pooling = "avg",
                toplayer_epochs = 0,
                trainable_layers = 0),
                filters={"high_grade": ["Y", "N"]},
                val_strategy='none',
                mixed_precision = True)

P.sources = ["CPTAC_BRCA"]
P.annotations = "../PROJECTS/HistoXGAN/cptac_all_annotations.csv"
model = '../PROJECTS/HistoxGAN/models/...enter location of trained model to evaluate'
dataset = P.dataset(tile_px = 512, tile_um = 400)

P.evaluate(model, 'high_grade', dataset=dataset, mixed_precision = True, save_predictions = True)    
```

### Attention-MIL Model Training
The following code can be used to train attention-MIL models (as used for visualization along prediction vs attention axes). 
```
from slideflow.model import build_feature_extractor
P = sf.Project('../PROJECTS/HistoXGAN/')
P.sources = ["TCGA_BRCA"]
P.annotations = "../PROJECTS/HistoXGAN/tcga_all_annotations.csv"
dataset = P.dataset(tile_px=512, tile_um=400)
dataset.build_index()
ctranspath = build_feature_extractor('ctranspath', tile_px=512)
P.generate_feature_bags(ctranspath, dataset, outdir = "../PROJECTS/HistoXGAN/bags/TCGA_BRCA/")
dataset = P.dataset(tile_px=512, tile_um=400, filters = {"high_grade": ["Y", "N"]})
dataset.build_index()
config = mil_config(model='attention_mil', aggregation_level = 'patient', lr = 0.0001, fit_one_cycle=True, wd=1e-5, epochs = 20)
P.train_mil(
    config=config,
    outcomes='high_grade',
    train_dataset=dataset,
    val_dataset=None,
    bags= "../PROJECTS/HistoXGAN/bags/TCGA_BRCA/"
)
```



