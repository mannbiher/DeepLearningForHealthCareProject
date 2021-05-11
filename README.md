# COVID-19 X-Ray Image Classification using Patched FLANNEL

This is source code repository for our research paper as part of final project
for UIUC graduate course CS598 Deep Learning for Healthcare. We propose a
combined model which takes patch based classification approach proposed by [Park
et al.](https://ieeexplore.ieee.org/document/9090149) and FLANNEL model proposed
by [Zhi Qiao et al](https://academic.oup.com/jamia/article/28/3/444/5943880). We
change base models from original FLANNEL to accept patch image for
classification and inference.

More details about our proposed model can be found in [project
report](doc/Project_Report.pdf),
[slides](https://docs.google.com/presentation/d/1F-7r4yMlsDHU1sbWj8aXkCuru2vJPpXnTpzeCDLBreM/edit?usp=sharing)
and [presentation](https://mediaspace.illinois.edu/media/t/1_atvzrp1d). Below
we provide description to setup, train and test original FLANNEL and patched
FLANNEL models.

## Authors

### Team Alchemists

| Name                         | NetId                 |
| ---------------------------- | --------------------- |
| Maneesh Kumar Singh          | mksingh4@illinois.edu |
| Raman Walwyn-Venugopal       | rsw2@illinois.edu     |
| Satish Reddy Asi             | sasi2@illinois.edu    |
| Srikanth Bharadwaz Samudrala | sbs7@illinois.edu     |

## Changes

To view changes from original source

```bash
cd src
git diff 1d593c9eb021c1353a030a934ef7dd8cf471def0 -- .
```

## Setup

The code for this repository has been developed on both Windows and Ubuntu.
However only Ubuntu is used to extensively test the models from end to end.
Therefore we would provide instructions to setup and run the code on Ubuntu.
Same instructions can be modified to run on Windows or Mac.

Create virtualenv with python >= 3.8.0 and activate it.

```bash
cd src
pip install -r requirements.txt
```

## Data Collection for FLANNEL

1. Download [Kaggle chest x-ray dataset] from
   <https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>.
2. Download COVID-19 chest x-ray dataset from
   <https://github.com/ieee8023/covid-chestxray-dataset/tree/78543292f8b01d5e0ed1d0e15dce71949f0657bb>.
   Notice the commit ID at the end. This commit is required to reproduce the
   results,
3. Unzip Kaggle and COVID-19 dataset in original_data folder in user directory.

```bash
unzip chest-xray-pneumonia.zip -d ~/original_data/
unzip master.zip -d ~/original_data/
```

## Metadata creation

Extract metadata from Kaggle and COVID-19 x-ray datasets by running below python
modules. These modules filter and extract metadata from x-ray images. The filter
includes selecting diseases from four classes COVID-19, Pneumonia virus,
Pneumonia bacteria and normal and selecting only PA and AP views. In addition to
class and view types, the metadata also contains image path. Metadata are stored
in Python pickle files.

```bash
# inside src folder
data_preprocess/get_covid_data_dict.py 
data_preprocess/get_kaggle_data_dict.py
```

## Segmentation

### Data Preparation

The required data can be downloaded from below mentioned sources. The data
retrieved must be put in a `data` folder as shown below.

```
data
  |- JSRT
    |- Image1
    |- Image 2
  |- SCR
    |- fold1/
    |- fold2
```

- Download images from their official sources and store in `data` folder
  - JSRT <http://db.jsrt.or.jp/eng.php>
  - SCR <https://www.isi.uu.nl/Research/Databases/SCR/download.php>
  - NLM(MC)
    <http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33>

### Training

- run train.py

```python
python train.py
```

This process should create a new model in output/model_v1.1pth

### Inference

This should be run after the Kaggle and GitHub datasets have been processed and
have had their appropriate pkl files generated. [formal_covid_dict_ap.pkl,
formal_kaggle_dict.pkl]

- copy model generated from training to the root of this folder

```python
cp output/model_v1.1pth ./model_v1.1pth
```

- run inference.py

```python
python inference.py
```

This process should generate a series of compressed Numpy files that contain a
Numpified version of the CXR with the mask applied. It also generates two new
pkl files [forml_covid_dict_ap.pkl.segmented.pkl,
formal_kaggle_dict.pkl.segmented.pkl]. These new pkl files annotate the original
entries with the path of the masked cxr npz file.

## Original FLANNEL

### Five Fold CV

Run below python module, to generate five-fold cross validation experiment data
from pickle files generated during metadata creation.

```bash
 python extract_exp_data_crossentropy \
    --out-dir ./data_preprocess/standard_data_multiclass_0922_crossentropy \
    --covid ./data_preprocess/formal_covid_dict_ap.pkl \
    --kaggle ./data_preprocess/formal_kaggle_dict.pkl
```

### Model Training

#### Base models training

To run model training for base FLANNEL, you can train each model directly using python
and provide all necessary arguments.

```bash
python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=200 --crop_size=299 -ck_n=50 --cv=cv1 -j=4
```

Or you could use [train_model.sh](src/train_model.sh) script in src folder,
which run all five folds using appropriate parameters for each model. This
scripts trains each model on 200 epochs and then evaluate the trained model.
There are total six models trained and it takes around 36 hours for training
and evaluation to finish on AWS p3.2xlarge instance.

```bash
./train_model.sh
```

#### Ensemble Model Training

Similary base model could be trained 



## Data Prepare
### Data Collect
1. , put them into original_data/covid-chestxray-dataset-master
2.
### Data Preprocess
1. extract data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract data from KCX: data_preprocess/get_kaggle_data_dict.py
3. train segmentation _(follow steps provided in [segmentation
   readme](data_preprocess/segmentation/README.md))_
4. perform segmentation on images data_preprocess/segmentation/inference.py
4. reorganize CCX&KCX data to generate 5-folder cross-validation expdata:
   data_preprocess/extract_exp_data_crossentropy.py



The code was tested on AWS EC2 p3.2xlarge instance type. You need to setup AWS
EC2 Role providing access to AWS S3 in order to backup model on S3. If you don't
want your model to be saved on S3 or if you are running the code on non-AWS
environment. You need to disable AWS S3 callbacks for backing up model on s3.

- In 'ensemble_step1.py
```diff
-      os.system("aws s3 sync {} {}".format(args.checkpoint, checkpoint_s3))
+      #os.system("aws s3 sync {} {}".format(args.checkpoint, checkpoint_s3))

```


## Model Training
### Base-modeler Learning
FLANNEL/ensemble_step1.py for 5 base-modeler learning [InceptionV3, Vgg19_bn,
ResNeXt101, Resnet152, Densenet161]

(E.g. python ensemble_step1.py --arch InceptionV3)

### ensemble-model Learning
FLANNEL/ensemble_step2_ensemble_learning.py


## Generate Plots

### Confusion Matrix

```bash
python FLANNEL/utils/result_plots.py explore_version_03/results
# toggle colormap to differentiate FLANNEL and patched FLANNEL
python FLANNEL/utils/result_plots.py patched_results/results togglecolor
```

### PR ROC Curves
```
python FLANNEL/utils/generate_pr_roc_curves.py
```

## References

- [FLANNEL paper](https://academic.oup.com/jamia/article/28/3/444/5943880)
- [Patch-based paper](https://ieeexplore.ieee.org/document/9090149)

## Acknowledgement

We have taken code provided by original authors of references paper as our
baseline.

- [FLANNEL github repository](https://github.com/qxiaobu/FLANNEL)
- [Patch-based model
  repository](https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets)
