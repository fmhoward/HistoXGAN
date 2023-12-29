# HistoXGAN
Accurate encoding of histologic images into StyleGAN space for discovery and explainability.
<img src='https://github.com/fmhoward/HistoXGAN/blob/main/histoxgan_architecture.png?raw=true'>

## Attribution
If you use this code in your work or find it helpful, please consider citing our preprint in <a href=''>bioRxiv</a>.
```

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
Training of HistoXGAN is fully integrated into the slideflow repository.

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

## HistoXGAN Applications
### Visualization of Histology Feature Space
Visualizing synthetic histology from a feature vector is easily performed with HistoXGAN; the included models allow for visualization of CTransPath and RetCCL feature vectors:
```
#Load the CTransPath HistoXGAN model
from slideflow.gan.stylegan3.stylegan3 import dnnlib, legacy, utils
with dnnlib.util.open_url('../FINAL_MODELS/CTransPath/snapshot.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

#Select a feature vector to visualize
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
