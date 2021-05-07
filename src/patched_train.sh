#!/usr/bin/env bash
set -x
# data_dir=./data_preprocess/standard_data_patched_0922_crossentropy
# python data_preprocess/extract_exp_data_crossentropy.py \
#     --out-dir $data_dir \
#     --covid ./data_preprocess/formal_covid_dict_ap.pkl.segmented.pkl \
#     --kaggle ./data_preprocess/formal_kaggle_dict.pkl.segmented.pkl
epochs=1
ck_n=50
workers=4
patches=1
for i in $(seq 1 1); do
    #python patched_flannel/entrypoint.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$workers
    # python patched_flannel/entrypoint.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers
    python patched_flannel/entrypoint.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test
done
# aws s3 sync explore_version_03/checkpoint s3://alchemists-uiuc-dlh-spring2021-us-east-2/patched_flannel_1/checkpoint/
