# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""
import os
import model
import torch


# Model
tag = 'v1.1'
filename_model = 'model_' + tag + '.pth'
dataset = 0
division_trainset = 1
threshold_partial = 1
partial_dataset = False
gamma = 0.5
num_channel = 1
ratio_dropout = 0.2
weight_bk = 0.5

# Directory 
dir_data_root = "~/data/"
dir_train_path = dir_data_root + "JSRT"
dir_test_path = dir_data_root + "JSRT"
dir_mask_path = ["/SCR/fold1/masks/heart", "/SCR/fold2/masks/heart", 
                "/SCR/fold1/masks/left lung", "/SCR/fold2/masks/left lung", 
                "/SCR/fold1/masks/right lung", "/SCR/fold2/masks/right lung"]
dir_checkpoint = "../checkpoint/"
dir_save = "../output/" 
dir_checkpoint = "./data_preprocess/segmentation/"
dir_save = os.path.expanduser("~/segmentation/output/")

# Network
num_masks = 4
num_network = 1
net = model.FCDenseNet(num_channel, num_masks, ratio_dropout) 
net_label = ['BG', 'Cardiac', 'Thorax(L)', 'Thorax(R)']

# Dataset
orig_height = 2048
orig_width = 2048
resize_height = 256 
resize_width = 256 
rescale_bit = 8 
# Resize image
post_resize = 1024

# CPU
num_worker = 8

# Train schedule
num_batch_train = 2
epoch_max = 100
threshold_epoch = 300
learning_rate = 1e-4

# Validation schedule
train_split = 0.8
valid_split = 0.92

# Test schedule
num_batch_test = 8

