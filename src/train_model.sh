#!/usr/bin/env bash
set -x
python data_preprocess/get_covid_data_dict.py
python data_preprocess/get_kaggle_data_dict.py
rm -rf './data_preprocess/standard_data_multiclass_0922_crossentropy'
python data_preprocess/extract_exp_data_crossentropy.py
for i in $(seq 1 2); do
    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=1 --crop_size=299 -ck_n=10 --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=1 --crop_size=299 -ck_n=10 --cv=cv$i --test --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=1 -ck_n=10 --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=1 -ck_n=10 --cv=cv$i --test --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=1 -ck_n=10 --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=1 -ck_n=10 --cv=cv$i --test --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=1 -ck_n=10 --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=1 -ck_n=10 --cv=cv$i --test --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=1 -ck_n=10 --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=1 -ck_n=10 --cv=cv$i --test --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i --gpu-id="0,1,2,3"
    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i --test --gpu-id="0,1,2,3"
done
