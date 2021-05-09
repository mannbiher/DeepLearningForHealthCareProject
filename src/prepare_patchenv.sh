#!/usr/bin/env bash
cd ~/DeepLearningForHealthCareProject/src
rm data_preprocess/pickle_05042021_latest.zip
git pull
unzip data_preprocess/patched_flannel_folds.zip -d data_preprocess/standard_data_patched_0922_crossentropy/
source ~/env
workon flannel
pip install -r requirements.txt