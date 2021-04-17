#!/usr/bin/env bash
#source ~/env
#workon flannel
set -x
python data_preprocess/get_covid_data_dict.py
python data_preprocess/get_kaggle_data_dict.py
rm -rf './data_preprocess/standard_data_multiclass_0922_crossentropy'
python data_preprocess/extract_exp_data_crossentropy.py