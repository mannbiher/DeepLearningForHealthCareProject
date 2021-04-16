#!/usr/bin/env bash
#source ~/env
#workon flannel
set -x
for i in $(seq 1 5); do
    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=1 --crop_size=299 -ck_n=10 --cv=cv$i
    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=1 --crop_size=299 -ck_n=10 --cv=cv$i --test
    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=1 -ck_n=10 --cv=cv$i
    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=1 -ck_n=10 --cv=cv$i --test
    python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=1 -ck_n=10 --cv=cv$i
    python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=1 -ck_n=10 --cv=cv$i --test
    python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=1 -ck_n=10 --cv=cv$i
    python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=1 -ck_n=10 --cv=cv$i --test
    python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=1 -ck_n=10 --cv=cv$i
    python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=1 -ck_n=10 --cv=cv$i --test
    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i
    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i --test
done
