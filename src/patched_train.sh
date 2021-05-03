cd /home/ubuntu/original_data -> For test data
cd /home/ubuntu/DeepLearningForHealthCareProject/src -> For git src from master branch
git pull
git checkout patched-flannel
python3 data_preprocess/get_covid_data_dict.py
python3 data_preprocess/get_kaggle_data_dict.py
pip3 install torch
pip3 install torchvision
python3 data_preprocess/segmentation/inference.py
python3 patched_flannel/extract_exp_data_crossentropy.py
python3 patched_flannel/classification/prep_classification_dataset.py
python3 patched_flannel/classification/classification_train.py 'cv1'
python3 patched_flannel/classification/classification_inference.py 'cv1'