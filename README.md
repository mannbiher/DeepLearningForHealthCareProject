# DeepLearningForHealthCareProject
Final project for UIUC graduate course CS598 Deep Learning for Healthcare

[FLANNEL paper](https://academic.oup.com/jamia/article/28/3/444/5943880)

[Patch by Patch paper](https://ieeexplore.ieee.org/document/9090149)


## Changes

To view changes from original source

```bash
cd src
git diff 1d593c9eb021c1353a030a934ef7dd8cf471def0 -- . :^PatchByPatch
```

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
#### Infrastructure
		Terraform
		Linux Bash

#### Technical Modules 

	flannel
		data_preprocess
		FLANNEL

	PatchByPatch
	Patched_flannel

#### Usage
    git pull https://github.com/mannbiher/DeepLearningForHealthCareProject.git
    cd DeepLearningForHealthCareProject
    git pull
    source ../env
    workon flannel
    chmod +x trainmodel.sh
    ./trainmodel.sh

#### Configuration (Infrastructure)

| Purpose  | Instance Type | GPU | vCPU | GPU Memory |
|----------|---------------|-----|------|------------|
| Development | AWS p2.xlarge | 1   | 4    | 12 GB      |
| Training | AWS p3.2xlarge | 1 | 8   | 16 GB      |

<!--					            GPU		vCPU		GPU Memory
    Training -	AWS p2.xlarge 	1		4			12 GB 
			    AWS p3.2xlarge	1		8			16 GB-->


#### Knows Issues
<!--[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=srasi1&layout=compact)](https://github.com/mannbiher/DeepLearningForHealthCareProject)-->

#### Troubleshooting
<!--![GitHub stats](https://github-readme-stats.vercel.app/api?username=srasi1&show_icons=true&theme=radical)-->

#### Technical challenges
#### References
#### Collaborators
* [Maneesh] (mailto:msingh4@illinois.edu)
* [Raman] (mailto:rsw2@illinois.edu)
* [Satish] (mailto:sasi2@illinois.edu)
* [Srikanth] (mailto:sbs7@illinois.edu)


<!--![](https://img.shields.io/badge/Maneesh_Kumar_Singh-Maneesh%20Kumar%20Singh-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)-->