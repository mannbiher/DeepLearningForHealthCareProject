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

# FLANNEL

## Update
TODO
fix below warnings
- /home/ubuntu/.virtualenvs/flannel/lib/python3.8/site-packages/torchvision/transforms/transforms.py:257: UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
  warnings.warn(
- FLANNEL/ensemble_step1.py:373: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  inputs, targets= torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
  

## Changes


### Ensemble Step1 volatile has no effect

```python
FLANNEL/ensemble_step2_ensemble_learning.py:375: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  inputs, targets= torch.autograd.Variable(inputs, volatile=True).float(), torch.autograd.Variable(targets)

```

```diff
-        inputs, targets= torch.autograd.Variable(inputs, volatile=True).float(), torch.autograd.Variable(targets)
+        inputs = inputs.float()
         # compute output
-        outputs = model(inputs)
-        loss = criterion(outputs, targets)
+        with torch.no_grad():
+            outputs = model(inputs)
+            loss = criterion(outputs, targets)
```

### Ensemble Step2 All tensors to be on same device

```python
Traceback (most recent call last):
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 420, in <module>
    main()
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 262, in main
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 315, in train
    loss = criterion(outputs, targets.type(torch.LongTensor).cuda())
  File "/home/neha/.virtualenvs/flannel/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/neha/DeepLearningForHealthCareProject/src/FLANNEL/models/proposedModels/loss.py", line 38, in forward
    loss = -1 * (1-p)**self.gamma * log_p * target_v * self.label_distri
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

```diff
         temp = torch.zeros(p.shape)
         if self.cuda_a:
           temp = temp.cuda()
+          if self.label_distri != None:
+            self.label_distri = self.label_distri.cuda()
         target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)

```




Views have been updated for Covid19 data
AP => AP and AP Erect

XRay data commit: 78543292f8b01d5e0ed1d0e15dce71949f0657bb

## Data Prepare
### Data Collect
1. Download CCX data: from https://github.com/ieee8023/covid-chestxray-dataset, put them into original_data/covid-chestxray-dataset-master
2. Download KCX data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, put them into original_data/chest_xray
### Data Preprocess
1. extract data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract data from KCX: data_preprocess/get_kaggle_data_dict.py
3. train segmentation _(follow steps provided in [segmentation readme](data_preprocess/segmentation/README.md))_
4. perform segmentation on images data_preprocess/segmentation/inference.py
4. reorganize CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy.py

## Model Training
### Base-modeler Learning
FLANNEL/ensemble_step1.py for 5 base-modeler learning [InceptionV3, Vgg19_bn, ResNeXt101, Resnet152, Densenet161]

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

## Acknowledgement

- [FLANNEL github repository](https://github.com/qxiaobu/FLANNEL)
- [Patch-based model repository](https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets)

#### Project Summary

#### Technologies & Tools

<!--![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)-->
![](https://img.shields.io/static/v1?label=OS&message=Linux&color=brightgreen)
![](https://img.shields.io/static/v1?label=Code&message=Python%203.8.0&color=brightgreen)
![](https://img.shields.io/static/v1?label=Infrastructure&message=AWS&color=brightgreen)
![](https://img.shields.io/static/v1?label=Infrastructure-Automation&message=Terraform&color=brightgreen)
![](https://img.shields.io/static/v1?label=Editor&message=VSCode&color=brightgreen)
![](https://img.shields.io/static/v1?label=Processor&message=GPU%20Tesla&color=brightgreen)
<!--color  yellowgreen brightgreen green yellow orange https://shields.io/ -->
###### Python Modules
```
  | - Python
	| - pandas
	| - numpy
	| - pytorch
	| - matplotlib
	| - opencsv
	| - torchvision
	| - scikit-learn
	| - tensorboard
	| - seaborn
```

###### Infrastructure
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
