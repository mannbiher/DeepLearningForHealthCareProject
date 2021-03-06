#!/usr/bin/env bash
set -x
# data_dir=./data_preprocess/standard_data_patched_0922_crossentropy
# python data_preprocess/extract_exp_data_crossentropy.py \
#     --out-dir $data_dir \
#     --covid ./data_preprocess/formal_covid_dict_ap.pkl.segmented.pkl \
#     --kaggle ./data_preprocess/formal_kaggle_dict.pkl.segmented.pkl
epochs=100
ck_n=50
workers=8
patches=100
for i in $(seq 1 5); do
	python patched_flannel/entrypoint.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$workers --in_memory
	python patched_flannel/entrypoint.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test --in_memory
	python patched_flannel/entrypoint.py --arch resnext101_32x8d --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --in_memory
	python patched_flannel/entrypoint.py --arch resnext101_32x8d --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test --in_memory
	python patched_flannel/entrypoint.py --arch resnet152 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --in_memory
	python patched_flannel/entrypoint.py --arch resnet152 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test --in_memory
	python patched_flannel/entrypoint.py --arch densenet161 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --in_memory
	python patched_flannel/entrypoint.py --arch densenet161 --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test --in_memory
	python patched_flannel/entrypoint.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers --in_memory --in_memory
	python patched_flannel/entrypoint.py --arch vgg19_bn --epochs=$epochs -ck_n=$ck_n --cv=cv$i -j=$workers -k=$patches --test --in_memory
done
# aws s3 sync patched_results/ s3://alchemists-uiuc-dlh-spring2021-us-east-2/patched_results_v3/ --acl bucket-owner-full-control
epochs=200
for i in $(seq 1 5); do
	python FLANNEL/ensemble_step2_ensemble_learning.py \
		--checkpoint ./patched_results/checkpoint \
		--results ./patched_results/results \
		--epochs=$epochs -ck_n=$ck_n \
		--data_dir='./patched_results/results/%s_20200407_patched_%s' --cv=cv$i -j=$workers --patched
	python FLANNEL/ensemble_step2_ensemble_learning.py \
		--checkpoint ./patched_results/checkpoint \
		--results ./patched_results/results \
		--epochs=$epochs -ck_n=$ck_n \
		--data_dir='./patched_results/results/%s_20200407_patched_%s' --cv=cv$i -j=$workers --test --patched
done

#aws s3 sync patched_results/ s3://alchemists-uiuc-dlh-spring2021-us-east-2/patched_results_v6/ --acl bucket-owner-full-control
