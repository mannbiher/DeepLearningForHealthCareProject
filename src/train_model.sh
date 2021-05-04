#!/usr/bin/env bash
#source ~/env
#workon flannel
set -x
epochs=200
ck_n=50
workers=6
for i in $(seq 1 5); do
#    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$workers
#    python FLANNEL/ensemble_step1.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$workers --test
#    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers
#    python FLANNEL/ensemble_step1.py --arch resnext101_32x8d --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --test
    python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers
    # python FLANNEL/ensemble_step1.py --arch resnet152 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --test
    python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers
    #python FLANNEL/ensemble_step1.py --arch densenet161 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --test
    #python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers
    #python FLANNEL/ensemble_step1.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --test
#    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i -j=$workers
#    python FLANNEL/ensemble_step2_ensemble_learning.py --epochs=1 -ck_n=50 --data_dir='./explore_version_03/results/%s_20200407_multiclass_%s' --cv=cv$i -j=$workers --test
done
aws s3 sync explore_version_03/checkpoint s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/
