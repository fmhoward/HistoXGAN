# HistoXGAN
Accurate encoding of histologic images into StyleGAN space for discovery and explainability


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

## Setup
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

## Slide Extraction
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

## HistoXGAN training
Training of HistoXGAN is fully integrated into the slideflow repository

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

## Tile-based Model Training / Evaluation
For training a model on TCGA for three fold cross validation of the held out set:

```
P = sf.Project('../PROJECTS/HistoXGAN/')
P.sources = ["TCGA_BRCA"]
P.annotations = "../PROJECTS/HistoXGAN/tcga_brca_complete.csv"
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
model = '../PROJECTS/HistoxGAN/models/...enter location of trained model to evaluate'
dataset = P.dataset(tile_px = 512, tile_um = 400)
P.evaluate(model, 'high_grade', dataset=dataset, mixed_precision = True, save_predictions = True)    
```
## Attention-MIL Model Training

## Hyperparameter Optimization
To perform hyperparameter optimization, run the model_training.py file with the following parameters (example given for 50 runs for hyperparameter optimization):
```
python model_training.py --hpsearch run --hpprefix DESIRED_PREFIX --hpstart 1 --hpcount 50
```

This code will automatically run hyperparameter optimization for the specified iterations. The following range of parameters are used for the hyperparameter search
```
Dropout: 0 - 0.5
Hidden Layer Width (dimension of fully connected layers after Xception backbone): 128 - 1024
Hidden Layers (number of fully connected hidden layers): 1 - 5
Learning Rate: 0.00001 - 0.001
Learning Rate Decay Steps (# of batches until applying learning rate decay): 128 - 1024
Learning Rate Decay Ratio (ratio with which to reduce the learning rate): 0 - 1
Learning Rate Decay (whether to decay learning rate): True / False
Loss (for linear loss, will train models on numerical recurrence score; for categorical loss will train on high / low cutoff of recurrence score): mean squared error, mean absolute error, sparse categorical cross entropy
Batch Size: 8 - 128
Augment (‘x’ performs horizontal flipping, ‘y’ performs vertical flipping, ‘r’ performs rotation, ‘j’ performs random JPEG compression): xyr, xyrj, xyrjb
Normalizer (whether to apply stain normalization): reinhard, none
L1 weight: 0 - 0.1
L1 (whether to perform L1 regularization): True, False
L2 weight: 0 - 0.1
L2 (whether to perform L2 regularization): True, False
L1 dense weight: 0 - 0.1
L1 dense (whether to perform l1 regularization to dense layers): True, False
L2 dense weight: 0 - 0.1
L2 dense (whether to perform l1 regularization to dense layers): True, False
```

Optimization is done using the <a href='https://pypi.org/project/smac/'>SMAC package</a>. Optimization is performed across two sets of three cross folds (listed in the tcga_brca_complete.csv file as CV3_odx85_mip and CV3_mp85_mip. These cross folds were chosen using <a href='https://github.com/fmhoward/PreservedSiteCV'>site preserved</a> splits to optimize the balance of high versus low risk OncotypeDx and MammaPrint recurrence scores. These splits can be regenerated, if desired, using the createCrossfoldsRS function in the model_analysis.py file. 

<img src="https://github.com/fmhoward/DLRS/blob/main/figures/bayesian.png?raw=true" width="600">


## Pathologic Model Training
To train models for tumor detection and region of interest annotation, run model_training.py with the following parameters:
```
python model_training.py -t --hpsearch read --hpprefix DESIRED_PREFIX --hpstart 0 --hpcount 50
```
	
This will search for the saved tile-level AUROC results from models stored within /PROJECTS/UCH_RS/models/ with the prefix DESIRED_PREFIX, identifying the hyperparameter combination with the highest tile-level AUROC.

Or, if you do not want to rerun hyperparameter optimization and would prefer to use stored hyperparameters from our optimization, they can automatically be  loaded as follows:
```
python model_training.py -t --hpsearch old
```

This command will train models on the entire TCGA dataset for prediction of MammaPrint and OncotypeDx scores, as well as three models trained for cross validation (again using the CV3_odx85_mip and CV3_mp85_mip headers to specify the folds). Predictions will be made for the held out 1/3 of the data, and saved in the /PROJECTS/UCH_RS/eval/ folder.
	
Several special keyword arguments can be provided to train models in an alternative fashion:
```
python model_training.py -t --hpsearch old -uf #The -uf or --use_filtered command will train models using tiles selected by the tumor-likelihood model instead of from pathologist annotations
python model_training.py -t --hpsearch old -tr #The -tr or --train_receptors command will train models using only HR+/HER2- patients from TCGA
python model_training.py -t --hpsearch old -rev #The -rev or --train_reverse command will train models on the UCMC dataset for validation in TCGA
```


Model training within this script is performed automatically using the slideflow train command. This could be performed manually outside of this automated script if desired as follows (example given for training 3 cross fold validated models within TCGA).

```
SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
SFP.sources = ["TCGA_BRCA_FULL_ROI"]
hp = hp_opt #uses saved optimal hyperparameters from optimization
odx_train_name = "GHI_RS_Model_NJEM.2004_PMID.15591335" #numerical OncotypeDx score in annotation file
odx_val_name = "odx85" #categorical representation of OncotypeDx score in annotation file
exp_label = "ODX_Final_BRCAROI"

SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI")) # need to setup the slideflow project for where tumor region of interest predictions are stored
SFP_TUMOR_ROI.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")
hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer) #assigns the tumor likelihood model (only used for validation)
SFP.train(exp_label=exp_label, outcome_label_headers=odx_train_name,  val_outcome_label_headers=odx_val_name, params = hp, filters = mergeDict(filters, {odx_val_name: ["H","L"]}), val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_odx85_mip", multi_gpu=True, save_predictions=True)

#Key arguments used in our slideflow training setup -
#exp_label - defaults to ODX_Final_BRCAROI - saved label for the experiment
#outcome_label_headers - defaults to the recurrence score header in the TCGA dataset - GHI_RS_Model_NJEM.2004_PMID.15591335 for Oncotype and Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860 for MammaPrint
#params - assigns model hyperparameters as described above
#filters - specify ranges allowed for specific columns in the annotations file
#val_strategy - set to k-fold-manual to allow k-fold cross validation with a consistent set of preserved site folds as generated above and included in our annotations file
#val_k_fold_header - specifies the header that assigns specific k-fold to each patient
#multi_gpu - allows training to utilize all available GPUs
#save_predictions - to save the outcome of model training
```
	
## Pathologic Model Prediction Generation
To validate the trained models in the UCMC dataset, run model_training.py with the -v flag:
```
model_training.py -v
```

This will generate patient level predictions from the pathologic model on the validation dataset, which are stored in the /PROJECTS/UCH_RS/eval folder. To illustrate how this validation is setup in the script (and thus how it could be applied to an external dataset):

```
#Set up a slideflow project referencing the validation dataset annotations and dataset source
SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
SFP.sources = ["UCH_BRCA_RS"]

#Set up the slideflow project for where tumor region of interest predictions are performed
SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
SFP_TUMOR_ROI.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")

#Load saved hyperparameters and assign the tumor likelihood model (weight_model) hyperparameter
hp = hp_opt 
hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer)

#Specifiy experiment names which were used for model training, and specify the column names used for outcome from training and validation
exp_label = "ODX_Final_BRCAROI"
odx_train_name = "GHI_RS_Model_NJEM.2004_PMID.15591335"
odx_val_name = "RSHigh"

#Finds the trained model within the SFP project folder (in this case /PROJECTS/UCH_RS/models/ folder)
m = find_model(SFP, exp_label, outcome = odx_train_name, epoch=hp.epochs[0])

#re-assign hyperparameters for categorical validation of a linear model
params = sf.util.get_model_config(m)
params["hp"]["loss"] = "sparse_categorical_crossentropy"
params["model_type"] = "categorical"
params["outcome_labels"] = {"0":"H","1":"L"}
params["onehot_pool"] = 'false'
sf.util.write_json(params, join(m, "params_eval.json"))

#Run OncotypeDx model validation
SFP.evaluate(model=m, outcome_label_headers=odx_val_name, save_predictions=True, model_config=join(m, "params_eval.json"))
```

## Replicating our Model Analysis	
To evaluate performance characteristics of the trained model, run model_analysis.py. This script will perform the following steps:
1. Load the TCGA dataset (from /PROJECTs/UCH_RS/tcga_brca_complete.csv) and TCGA pathology model predictions (from /PROJECTS/UCH_RS/eval/), and compute the clinical nomogram results for each patient. In the case of the MammaPrint model, the clinical predictions will be generated using a logistic regression fit on NCDB (saved in the installation directory as NCDB2017.csv).
2. Fit three logistic regression models on 2/3 of the data using out-of-sample pathology predictions and clinical nomogram predictions. Make predictions with logistic regression model on remaining 1/3 of the data. 
3. Detects thresholds to use for a rule-out (95% sensitivity) model in each of the three cross folds to generate an average threshold for validation.
4. Generate ROC curves and fit survival models on TCGA using the predictions from the pathologic, clinical, and combined models. Evaluate the rule-out threshold performance.
5. Fit a combined model logistic regression using the average coefficients of the three logistic regressions fit in TCGA
6. Loads the UCMC dataset (from /PROJECTS/UCH_RS/uch_brca_complete.csv) and UCMC pathology model predictions (from /PROJECTS/UCH_RS/eval/), and computes the clinical nomogram results for each patient, as well as the results from the combined logistic regression.
7. Generates ROC curves and fit survival models on UCMC using the predictions from the pathologic, clinical, and combined models. Evaluate the rule-out threshold performance.
8. Computes correlation between model predictions and grade, necrosis, and lymphovascular invasion.

<img src="https://github.com/fmhoward/DLRS/blob/main/figures/analysis.png?raw=true" width="600">
	
Plots are saved to the root directory:  
ROC Curves <outcome>.png - which has ROC curves for the TCGA / UCMC datasets for all three models for the Oncotype or MammaPrint outcome  
Prognostic Plots <outcome>.png - which plots Kaplan Meier curves for high versus low risk patients identified by the high sensitivity thresholds for the clinical nomogram and the combined model in the validation dataset  
Prognostic Plots TCGA <outcome>.png - which plots Kaplan Meier curves for high versus low risk patients within TCGA  	
Correlation <outcome>.png - which plots linear correlation between the model and true recurrence score results  
Prognostic Comparison.png - comparison between the prognostic outcomes of the TCGA and UCMC datasets  
HP Search.png - a plot of hyperparameter evolution over Bayesian optimization  

Baseline demographics are saved in the root directory:  
UCMC MammaPrint Cohort.xlsx  
UCMC Oncotype Cohort.xlsx  
TCGA Cohort.xlsx  
TCGA HRHER2 Cohort.xlsx  

All performance metrics are saved in the root directory:  
predictions_<outcome>.csv - contains main predictions including AUROC/AUPRC and prognostic performance of the specified model  	
correlate_pathparams.csv - contains the correlation coefficients for model predictions and pathologic parameters  

Several special parameters can be provided, in particular the -s command will use the saved predictions allowing easy replication of our analysis without running model_training.py:
```
python model_analysis.py -s #The -s or --saved command will use saved pathologic model predictions from /PROJECT/saved_results/ and embedded in the uch_brca_complete.csv and tcga_brca_complete.csv files (rather then reloading the predictions from the /PROJECTS/UCH_RS/eval/ directory).
python model_analysis.py -uf #The -uf or --use_filtered command will train models using tiles selected by the tumor-likelihood model instead of from pathologist 
annotations
python model_analysis.py -tr #The -tr or --train_receptors command will train models using only HR+/HER2- patients from TCGA
python model_analysis.py -rev #The -rev or --train_reverse command will train models on the UCMC dataset for validation in TCGA
```

## Testing Predictions on New Patients
To generate predictions on an external dataset (i.e. to test this on new patients), the necessary columns to include in the annotations file are:
```
patient - patient identifier
slide - slide file name
Age - numeric age in years - for calculation of clinical nomogram
hist_type - histologic subtype - Ductal, Lobular, D&L, or Other - for calculation of clinical nomogram
grade - 1, 2, or 3 - for calculation of clinical nomogram
tumor_size - tumor size measured in mm - for calculation of clinical nomogram
PR  - 'Pos', or 'Neg', for calculation of clinical nomogram
```

This should be followed by setting up the datasets.json as above with a new dataset reflecting the slide directory of your new images, and the location you would like to store tfrecords. 

The model_training file can be used to extract slides and generate predictions for this new dataset:
```
python model_training.py --extract --annotation <annotation.csv file name, assumed to be in /PROJECTS/UCH/> --source <dataset name specified in dataset.json> --experiment_label <unique label for experiment; duplicate models with same prefix leads to errors>
python model_training.py --validate --annotation <annotation.csv file name, assumed to be in /PROJECTS/UCH/> --source <dataset name specified in dataset.json> --experiment_label <label>
```
	
The model_analysis file can use the annotation file (with proper annotations for the nomogram) along with the pathologic predictions from model_training.py to generate predictions on patients with unknown Oncotype score -
```
python model_analysis.py -pred --outcome <RS for Oncotype or MP for MammaPrint> --dataset <name of CSV file with annotations in the UCH_RS folder>  --experiment_label <label>
```

These predictions will be saved in the project root as "<dataset>_predictions.csv"; with columns including percent_tiles_positive_0, ten_score, and comb - corresponding to the numeric predictions of the pathologic, clinical, and combined models. Columns percent_tiles_positive_0_thresh, ten_score_thresh, and comb_thresh - correspond to a binary of whether a patient was predicted high risk (1) or low risk (0) using the high sensitivity threshold.

To make new predictions using our frozen trained models for this analysis, please <a href='doi.org/10.5281/zenodo.7490381'>download the trained models from Zenodo</a> and extract the zip into the PROJECTS folder.


## Model interpretation
To view heatmaps from trained models, run model_training.py with the --heatmaps_tumor_roi for tumor likelihood predictions or --heatmaps_odx for Oncotype model predictions, and specify 'TCGA' or 'UCH' depending on which dataset you want to generate heatmaps for:
```
python model_training.py --heatmaps_tumor_roi TCGA
python model_training.py --heatmaps_odx TCGA
```
<img src="https://github.com/fmhoward/DLRS/blob/main/figures/heatmaps.png?raw=true" width="600">
