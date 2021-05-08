from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
import os.path
from classification import header
from PIL import Image
from torch.utils import data
from torchvision import models, transforms
import torch
import random
from classification.utils import (
    augmentation, parse_data_dict,
    data_transforms
)


class COVID_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dim=(224, 224), n_channels=3, n_classes=4, mode='train', opts=None):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        if mode == 'val':
            mode = 'valid'
        self.data_dir = opts.data % mode
        self.img_size = opts.crop_size
        self.image_paths, self.label_list, self.info_list = parse_data_dict(
            self.data_dir, opts.in_memory)

        # self.labels = os.listdir(self.data_dir) # COVID, Bacteria, Virus, TB, Normal

        # self.total_images_dic = {}

        print('Generator: %s' % self.mode)
        print('A total of %d image data were generated.' %
              len(self.image_paths))

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        self.n_data = len(self.image_paths)
        self.classes = [i for i in range(n_classes)]
        self.imgs = self.image_paths

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        X, y = self.__data_generation(index)
        return X, y

    def __data_generation(self, index):
        # X : (n_samples, *dims. n_channels)
        'Generates data containing batch_size samples'
        # Generate data & Store sample
        # Assign probablity and parameters

        rand_p = random.random()

        X_masked = np.load(self.image_paths[index])['image']

        h_whole = X_masked.shape[0]  # original w
        w_whole = X_masked.shape[1]  # original h

        # print('size of patch', h_whole, w_whole)

        non_zero_list = np.nonzero(X_masked)

        # random non-zero row index
        non_zero_row = random.choice(non_zero_list[0])
        # random non-zero col index
        non_zero_col = random.choice(non_zero_list[1])

        X_patch = X_masked[int(max(0, non_zero_row - (self.img_size / 2))):
                           int(min(h_whole, non_zero_row + (self.img_size / 2))),
                           int(max(0, non_zero_col - (self.img_size / 2))):
                           int(min(w_whole, non_zero_col + (self.img_size / 2)))]

        # print('size', X_patch.shape)

        X_patch_img = self.data_transforms(np.array(augmentation(
            Image.fromarray(X_patch), rand_p=rand_p, mode=self.mode)))
        X_patch_img_ = np.squeeze(np.asarray(X_patch_img))

        X_patch_1 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_2 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_3 = np.expand_dims(X_patch_img_, axis=0)

        X_ = np.concatenate((X_patch_1, X_patch_2, X_patch_3), axis=0)
        X = torch.from_numpy(X_)

        # Store classes
        y = self.label_list[index]

        return X, y
