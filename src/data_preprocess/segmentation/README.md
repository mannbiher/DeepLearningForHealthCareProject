# Segmentation

## Data Prep

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

- __Option 1:__ Download zipped datasets from [s3](https://s3.console.aws.amazon.com/s3/buckets/alchemists-uiuc-dlh-spring2021-us-east-2?region=us-east-2&prefix=patch-by-patch/data/&showversions=false) and store in `data` folder

- __Option 2:__ Download images from their official sources and store in `data` folder
    - JSRT               http://db.jsrt.or.jp/eng.php
    - SCR                https://www.isi.uu.nl/Research/Databases/SCR/download.php
    - NLM(MC)            http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33


## Training

- run train.py
```
python train.py
```

This process should create a new model in output/model_v1.1pth

## Inference

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