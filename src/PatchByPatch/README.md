
# Alchemist Prefix

__Requirements__
- Nvidia GPU
- Python3

__Setup__
- Create virtual environment
```
python -m venv venv
```

- activate virtual environment

- install packages
```
python -m pip install -r requirements.txt
```

- download patch by patch data from s3 database https://msingh.signin.aws.amazon.com/console and store in `data` folder

- unzip JSRT and SRC files so the directory structure looks like this
```
JSRT
  |- Image1
  |- Image 2
SCR
  |- fold1/
  |- fold2
```

__Segmentation__
- ensure you are in activated virtual environment with installed packages
- enter segmentation directory
```
cd segmentation
```
- run train.py
```
python train.py
```
- run inference.py
```
python inference.py
```

__Classification__
- prep classification dataset
```
python prep_classification_dataset.py
```
- run training
```
python classification_train.py
```
- run inference
```
python classification_inference.py
```

__Notes__
- classification
  - classification/header/repeat is the number of patches
  - the utils/customloader __getitem__ function calls __data_generation__ which imports the numpified image and mask, then creates a masked version of the image, then creates a patch from the numpified patch
  - the classification/infernece.py repeats by number of patches it calls getItem for
- how do we add segmentation to FLANNEL??
  - IDEA 1: run segmentation of all images first
    - will need to run segmentation on all images in data pre-processing
    - need to modify data pre-processing step to store image and mask information along with the labels
    - need to modify data loader to behave more like customloader __data_generation
  - IDEA 2: run segmentation of image in data loader
    - will need to load instance of segmentation model when data loader initializes
    - should we modify the __get_item__ to return a patch?
- how do we add K-Patch model to FLANNEL??


# Summary
-------
This is the Github repository for "Deep Learning COVID-19 on CXR using Limited Training Data Sets".

![methods](https://user-images.githubusercontent.com/39784965/81655488-3d5de380-9471-11ea-8f4b-b18e5fda7d08.png)

The proposed method consists of two-step approaches, namely "making segmentation mask" and "classification".
Each tasks are uploaded in seperated folders.

# Instructions
-------
To make segmentation masks, you can use codes included in "segmentation folder".
After making segmentation masks, generated masks as well as preprocessed x-ray images should be moved to "classification folder".

The detailed instruction is provided below.



# Dataset
------
We provide the links for COVID-19 public datasets, which are used in our recent publication, "Deep Learning COVID-19 on CXR using Limited Training Data Sets".

https://ieeexplore.ieee.org/document/9090149


These public CXR datasets are available without any restrction by downloading in below links.​

[19] JSRT                http://db.jsrt.or.jp/eng.php

[20] SCR                 https://www.isi.uu.nl/Research/Databases/SCR/download.php

[21] NLM(MC)            http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33

[22] Pneumonia          https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

[23] COVID-19           https://github.com/ieee8023/covid-chestxray-dataset



Please reference the metadata for your information.

https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets/blob/master/metadata.xls


You can also find corresponding links of each dataset in the page as below.

https://bispl.weebly.com/covid-19-data-link.html



# Segmentation
------
This is detailed instructions for the first step, segmentation.

You can access requried codes in "segmentation" folder.

Download JSRT/SCR training dataset in "data" folder with corresponding repository name as described in "segmentation/header.py".

For training, just run "segmentation/train.py".

Download inference datasets in "data" folder and rename each dataset folder as its corresponding class name. Modify "folder_list" in "segmentation/inference.py" with each class name. For further information of inference dataset, please refer Table 2 in the manuscript and "segmentation/metadata.xls".

For inference, just run "segmentation/inference.py".

The segmented mask (.mask.npy) and corresponding image (.image.npy) will be saved in "output" folder.


# Classification
------
This is detailed instructions for the second step, classification.

After creating preprocessed image file (.image.npy) and mask (.mask.npy), seperate and divide these into train, val, and test folders. The name of preprocessed image file and mask should be the same, except for the file format.

In our experiments, we randomly divided dataset into 0.7, 0.1, and 0.2 ratio.
Each folder (train, val, test) should contain daughter folders for labels (normal, bacteria, TB, COVID, virus) for dataloader get labels from image folder.

You can access requried codes in "classification" folder.

Before the training, you should specify test name and training options in "classification/header.py".

For training, just run "classification/train.py".

For inference, just run "classification/inference.py".

Finally, to get probabilistic Grad-CAM based saliency map, run "classification/visualize_cnn.py".


# Publication
-------
To cite this, 

@ARTICLE{9090149,
  author={Y. {Oh} and S. {Park} and J. C. {Ye}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Deep Learning COVID-19 Features on CXR using Limited Training Data Sets}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},}
  
https://ieeexplore.ieee.org/document/9090149
