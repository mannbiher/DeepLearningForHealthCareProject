"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.data

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}
    
def __add_noise(img, mean=0., std=1., size = None):
    return img + torch.randn(size) * std + mean
#    return img

def get_transform_augumentation(opt, params=None, transform_type = None, grayscale=False, method=Image.BICUBIC, convert=True, run_type = 'train'):
    
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if run_type == 'train':
      if transform_type == 'fz_horizontal':
          transform_list += [transforms.RandomHorizontalFlip()]
      elif transform_type == 'fz_vertical':
          transform_list += [transforms.RandomVerticalFlip()]
      elif transform_type == 'random_crop1':
          transform_list += [transforms.RandomCrop(opt.crop_size, pad_if_needed=True)]
      elif transform_type == 'random_crop2':
          transform_list += [transforms.RandomCrop(opt.crop_size, pad_if_needed=True)]
      elif transform_type == 'scale_0.5':
          transform_list += [transforms.RandomResizedCrop(opt.crop_size, scale=(0.08, 0.51), ratio=(1., 1.))]
      elif transform_type == 'scale_2':
          transform_list += [transforms.RandomResizedCrop(opt.crop_size, scale=(1.00, 2.00), ratio=(1., 1.))]

      if 'resize' in opt.preprocess:
          osize = [opt.load_size, opt.load_size]
          transform_list.append(transforms.Resize(osize, method))
      elif 'scale_width' in opt.preprocess:
          transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

      if 'crop' in opt.preprocess:
          if params is None:
              transform_list.append(transforms.RandomCrop(opt.crop_size))
          else:
              transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

      if convert:
          transform_list += [transforms.ToTensor()]
          osize = [opt.crop_size, opt.crop_size]
          if transform_type == 'gaussian_0_1':
              transform_list += [transforms.Lambda(lambda img: __add_noise(img, 0., 1., osize))]
          if transform_type == 'gaussian_05_1':
              transform_list += [transforms.Lambda(lambda img: __add_noise(img, 0.5, 1., osize))]
          if transform_type == 'gaussian_50_1':
              transform_list += [transforms.Lambda(lambda img: __add_noise(img, -0.5, 1., osize))]

          if grayscale:
              transform_list += [transforms.Normalize((0.5,), (0.5,))]
          else:
              transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
      if 'resize' in opt.preprocess:
          osize = [opt.load_size, opt.load_size]
          transform_list.append(transforms.Resize(osize, method))
      if 'crop' in opt.preprocess:
          transform_list.append(transforms.Lambda(lambda img: __crop(img, (opt.load_size/2, opt.load_size/2), opt.crop_size)))
      if convert:
          transform_list += [transforms.ToTensor()]
          if grayscale:
              transform_list += [transforms.Normalize((0.5,), (0.5,))]
          else:
              transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
       
    return transforms.Compose(transform_list)

  
def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, run_type = 'train'):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if run_type == 'train':
      if 'resize' in opt.preprocess:
          osize = [opt.load_size, opt.load_size]
          transform_list.append(transforms.Resize(osize, method))
      elif 'scale_width' in opt.preprocess:
          transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

      if 'crop' in opt.preprocess:
          if params is None:
              transform_list.append(transforms.RandomCrop(opt.crop_size))
          else:
              transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

      if opt.preprocess == 'none':
          transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

      if not opt.no_flip:
          if params is None:
              transform_list.append(transforms.RandomHorizontalFlip())
          elif params['flip']:
              transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

      if convert:
          transform_list += [transforms.ToTensor()]
          if grayscale:
              transform_list += [transforms.Normalize((0.5,), (0.5,))]
          else:
              transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
      if 'resize' in opt.preprocess:
          osize = [opt.load_size, opt.load_size]
          transform_list.append(transforms.Resize(osize, method))
      if 'crop' in opt.preprocess:
          transform_list.append(transforms.Lambda(lambda img: __crop(img, ((opt.load_size-opt.crop_size)/2, (opt.load_size-opt.crop_size)/2), opt.crop_size)))
      if convert:
          transform_list += [transforms.ToTensor()]
          if grayscale:
              transform_list += [transforms.Normalize((0.5,), (0.5,))]
          else:
              transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
