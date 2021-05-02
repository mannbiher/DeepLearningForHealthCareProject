from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import pickle
import os, os.path
import header
from PIL import Image
from torch.utils import data
import torch
import glob
import random
from utils import augmentation
import utils as utils


class COVID_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dim=(224, 224), n_channels=3, n_classes=4, mode='train', cv='cv1'):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        pkl_file_name = 'exp_'+mode+'_list_'+cv+'.pkl'
        if self.mode == 'train':
            self.read_pkl = './classification_data//standard_data_multiclass_0922_crossentropy/'+pkl_file_name
        elif self.mode == 'val':
            self.read_pkl = './classification_data//standard_data_multiclass_0922_crossentropy/'+pkl_file_name
        elif self.mode == 'test':
            self.read_pkl = './classification_data//standard_data_multiclass_0922_crossentropy/'+pkl_file_name

        #self.labels = os.listdir(self.data_dir) # COVID-19, pneumonia_bacteria, pneumonia_virus, normal, Ignore Unknown?

        self.total_images_dic = {}
        self.total_masks_dic = {}
        tuples_list = pickle.load(open(self.read_pkl, ‘rb’))
        for tuples in tuples_list:
            y_label = tuples[3]
            img_path = tuples[0]
        #for label in self.labels:

        #    npy_dir = self.data_dir + label

        #    if label == 'normal':
        #        y_label = 0
        #    elif label == 'pneumonia_virus':
        #        y_label = 1
        #    elif label == 'pneumonia_bacteria':
        #        y_label = 2
        #    elif label == 'COVID-19':
        #        y_label = 3

            images_list = glob.glob(img_path + '/*.image.npy')
            for image in images_list:
                self.total_images_dic[image] = y_label

            masks_list = glob.glob(img_path + '/*.mask.npy')
            for mask in masks_list:
                self.total_masks_dic[mask] = y_label

        print('Generator: %s' %self.mode)
        print('A total of %d image data were generated.' %len(self.total_images_dic))

        self.data_transforms = utils.data_transforms

        self.n_data = len(self.total_images_dic)
        self.classes = [i for i in range(n_classes)]
        self.imgs = self.total_images_dic

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        X, y = self.__data_generation(index)
        return X, y

    def __data_generation(self, index):

        'Generates data containing batch_size samples' # X : (n_samples, *dims. n_channels)
        # Generate data & Store sample
        # Assign probablity and parameters

        rand_p = random.random()

        # X_img
        X_whole = Image.fromarray(np.load(list(self.total_images_dic.keys())[index])).resize((header.resize, header.resize))
        X_whole = np.asarray(X_whole)

        h_whole = X_whole.shape[0] # original w
        w_whole = X_whole.shape[1] # original h

        X_whole_mask = Image.fromarray(np.load(list(self.total_images_dic.keys())[index].split('.image.npy')[0] + '.mask.npy')).resize((
                                                                                                                                       header.resize, header.resize))
        X_whole_mask = np.round(np.asarray(X_whole_mask))

        X_masked = np.multiply(X_whole, X_whole_mask)

        non_zero_list = np.nonzero(X_masked)

        non_zero_row = random.choice(non_zero_list[0]) # random non-zero row index
        non_zero_col = random.choice(non_zero_list[1]) # random non-zero col index

        X_patch = X_masked[int(max(0, non_zero_row - (header.img_size / 2))):
                           int(min(h_whole, non_zero_row + (header.img_size / 2))),
                  int(max(0, non_zero_col - (header.img_size / 2))):
                  int(min(w_whole, non_zero_col + (header.img_size / 2)))]

        X_patch_img = self.data_transforms(augmentation(Image.fromarray(X_patch), rand_p=rand_p, mode=self.mode))
        X_patch_img_ = np.squeeze(np.asarray(X_patch_img))

        X_patch_1 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_2 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_3 = np.expand_dims(X_patch_img_, axis=0)

        X_ = np.concatenate((X_patch_1, X_patch_2, X_patch_3), axis=0)
        X = torch.from_numpy(X_)

        # Store classes
        y = list(self.total_images_dic.values())[index]

        return X, y