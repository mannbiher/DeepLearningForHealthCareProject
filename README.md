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