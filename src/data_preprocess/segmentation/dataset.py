import header
import torch
from torchvision.transforms import functional as TF
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image

import csv
import shutil

class SegmentationDataset(Dataset):

    def __init__(self, formal_dict):

        self.formal_dict = formal_dict
        self.data_len = len(self.formal_dict)
        self.ids = list(self.formal_dict.keys())

        #if not os.path.isdir(image_path):
        #    raise RuntimeError("Dataset not found or corrupted. DIR: " + image_path)

        #for sample in self.images:
        #    self.ids.append(sample.replace(image_path, ''))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images = self.get_original(index).astype('float32')
        original_image_size = np.asarray(images.shape)
        og_id = self.ids[index]
        img_name = list(self.formal_dict[og_id]['image_dict'].keys())[0]
        class_dict = self.formal_dict[og_id]['class']
        class_dict_keys = list(class_dict.keys())
        class_dict_values = list(class_dict.values())
        class_index = list(filter(lambda i: i if i > 0 else -1, class_dict_values))[0]
        if class_index > 0 :
            class_name = class_dict_keys[class_index]
        else:
            class_name = 'Uknown'
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size, 'im_name':img_name, 'im_class':class_name}

    def get_original(self, index):
        og_id = self.ids[index]
        img_name = list(self.formal_dict[og_id]['image_dict'].keys())[0]
        img_data = self.formal_dict[og_id]['image_dict'][img_name]
        img_path = img_data['path']
        images = Image.open(img_path)
        if (np.asarray(images).max() <= 255):
            images = images.convert("L")
        images = np.asarray(images)

        line_center = images[int(images.shape[0]/2):, int(images.shape[1]/2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        images = pre_processing(images, flag_jsrt=0)

        return images

def pre_processing(images, flag_jsrt = 10):

    num_out_bit = 1<<header.rescale_bit
    num_bin = images.max() + 1
    hist, bins = np.histogram(images.flatten(), num_bin, [0, num_bin])
    cdf = hist_specification(hist, num_out_bit, images.min(), num_bin, flag_jsrt)
    images = cdf[images].astype('float32')

    return images

def hist_specification(hist, bit_output, min_roi, max_roi, flag_jsrt):

    cdf = hist.cumsum()
    cdf = np.ma.masked_equal(cdf, 0)

    # hist sum of low & high
    hist_low = np.sum(hist[:min_roi+1]) + flag_jsrt
    hist_high = cdf.max() - np.sum(hist[max_roi:])

    # cdf mask
    cdf_m = np.ma.masked_outside(cdf, hist_low, hist_high)

    # build cdf_modified
    if not (flag_jsrt):
        cdf_m = (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    else:
        cdf_m = (bit_output-1) - (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m.astype('float32'), 0)

    # gamma correction
    cdf = pow(cdf/(bit_output-1), header.gamma) * (bit_output-1)

    return cdf


def one_hot(x, class_count):

    return torch.eye(class_count)[:, x]


def create_folder(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_size_id(idx, size, case_id, dir_label):

    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    dir_results = [case_id + case_id + '_' + j + '.png' for j in dir_label]

    return original_size_w_h, case_id, dir_results