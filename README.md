# DeepLearningForHealthCareProject
Final project for UIUC graduate course CS598 Deep Learning for Healthcare

[FLANNEL paper](https://academic.oup.com/jamia/article/28/3/444/5943880)

[Patch by Patch paper](https://ieeexplore.ieee.org/document/9090149)


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

## Segmentation

### Data Prep

There are two soures where the images can be retrieved from, s3 pre-organized in a zip or from the official sources themselves. The data retrieved must be put in a `data` folder as shown below.

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
    - JSRT               http://db.jsrt.or.jp/eng.php
    - SCR                https://www.isi.uu.nl/Research/Databases/SCR/download.php
    - NLM(MC)            http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33


### Training

- run train.py
```
python train.py
```

This process should create a new model in output/model_v1.1pth

### Inference

This should be run after the kaggle and github datasets have been processed and have had their appropriate pkl files generated. [formal_covid_dict_ap.pkl, formal_kaggle_dict.pkl]

- copy model generated from training to the root of this folder
```
cp output/model_v1.1pth ./model_v1.1pth
```
- run inference.py
```
python inference.py
```

This process should generate a series of compressed numpy files that contain a numpified version of the CXR with the mask applied. It also generates two new pkl files [forml_covid_dict_ap.pkl.segmented.pkl, formal_kaggle_dict.pkl.segmented.pkl]. These new pkl files annotate the original entries with the path the the masked cxr npz file.



## Setup

Create virtualenv with python >= 3.8.0 and activate it.
```bash
cd src
pip install -r requirements.txt
```

The code was tested on AWS EC2 p3.2xlarge instance type. You need to setup AWS
EC2 Role providing access to AWS S3 in order to backup model on S3. If you don't
want your model to be saved on S3 or if you are running the code on non-AWS
environment. You need to disable AWS S3 callbacks for backing up model on s3.

```bash
# TODO
```

To train and evaluate the model. This scripts trains each model on 200 epochs and
then evaluate the trained model. There are total six models trained and it takes
around 36 hours for training and evaluation to finish on AWS p3.2xlarge instance.


```bash
./train_model.sh
```



## Generate Plots

### Confusion Matrix

```bash
python FLANNEL/utils/result_plots.py explore_version_03/results
# toggle colormap to differentiate FLANNEL and patched FLANNEL
python FLANNEL/utils/result_plots.py patched_results/results togglecolor
```

## Acknowledgement

- [FLANNEL github repository](https://github.com/qxiaobu/FLANNEL)
- [Patch-based model repository](https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets)

#### Project Summary

#### Technologies & Tools

<!--![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)-->
![](https://img.shields.io/static/v1?label=OS&message=Linux&color=yellowgreen)
![](https://img.shields.io/static/v1?label=Code&message=Python%203.8.0&color=brightgreen)
![](https://img.shields.io/static/v1?label=Infrastructure&message=AWS&color=green)
![](https://img.shields.io/static/v1?label=Infrastructure-Automation&message=Terraform&color=yellowgreen)
![](https://img.shields.io/static/v1?label=Editor&message=VSCode&color=yellow)
![](https://img.shields.io/static/v1?label=Processor&message=GPU%20Tesla&color=orange)

    Core part
	Python
	Libraries
		pandas
		numpy
		pytorch
		matplotlib
		opencsv
		torchvision
		scikit-learn
		tensorboard
		seaborn
#### Infrastructure
		Terraform
		Linux Bash

#### Project Modules 

##### data_preprocess
Segmentation network acts as the post data preprocessing step for PatchByPatch model. Segmentation model extracts lung and heart contours from the chest radiography images. 
Segmentation module uses fully convolutional DenseNet103 to perform semantic segmentation

data_preprocess model splits data into K folds and provides all the numbers for each of these K-folds, types of datasets and classes of images

Responsible for organizing data into different classes and extracting lung contours from CXR images

##### FLANNEL

The original FLANNEL code that was the foundation of our improvements

##### patched_flannel

The patch-based models that we've developed to be used in the patched_flannel

#### Usage

```shell script:
git pull https://github.com/mannbiher/DeepLearningForHealthCareProject.git
source ./env
workon flannel
cd DeepLearningForHealthCareProject
git pull
cd src
pip install -r requirements.txt
chmod +x trainmodel.sh
./trainmodel.sh
```

#### Configuration (Infrastructure)

| Purpose     | Instance Type  | GPU | vCPU | GPU Memory |
| ----------- | -------------- | --- | ---- | ---------- |
| Development | AWS p2.xlarge  | 1   | 4    | 12 GB      |
| Training    | AWS p3.2xlarge | 1   | 8    | 16 GB      |


#### Known Issues

<!-- Can we update google doc with all the known issues -->

#### Troubleshooting

<!-- Can we update google doc with all the known issues -->

#### Technical challenges
<!-- Can we update google doc with all the known issues -->
#### References
#### Collaborators
* [Maneesh](mailto:msingh4@illinois.edu)
* [Raman](mailto:rsw2@illinois.edu)
* [Satish](mailto:sasi2@illinois.edu)
* [Srikanth](mailto:sbs7@illinois.edu)
