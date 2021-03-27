#!/usr/bin/env bash
python data_preprocess/get_covid_data_dict.py
python data_preprocess/get_kaggle_data_dict.py
python data_preprocess/extract_exp_data_crossentropy.py
python FLANNEL/ensemble_step1.py --arch inception_v3
python FLANNEL/ensemble_step1.py --arch vgg19_bn
python FLANNEL/ensemble_step1.py --arch resnext101_32x8d
python FLANNEL/ensemble_step1.py --arch resnet152
python FLANNEL/ensemble_step1.py --arch densenet161