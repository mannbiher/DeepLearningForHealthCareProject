import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_augumentation
#from data.image_folder import make_dataset, make_label_dict
from data.image_folder import parse_data_dict
from models.segmentation.model import FCDenseNet as segmentation_model
from PIL import Image
import numpy as np
import torch



# add data augumentation 

class XrayDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, run_type = 'train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_dir = opt.data%run_type  # get the image data directory
        self.image_paths, self.label_list, self.info_list = parse_data_dict(self.data_dir)  # get image paths
        self.run_type = run_type

        # setup segmentation
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # initialize model
        segmentation_net = segmentation_model(1, 4, 0.2)
        segmentation_model_path = os.path.join('models', 'segmentation', 'model_v1.1.pth')
        segmentation_checkpoint = torch.load(segmentation_model_path)
        segmentation_net.load_state_dict(segmentation_checkpoint['model_state_dict'])
        # load model on to torch device/GPU
        segmentation_net.to(device)
        self.segmentation_net = segmentation_net

        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        xray_path = self.image_paths[index]
        xray = Image.open(xray_path).convert('L')
        label = self.label_list[index]
        info = self.info_list[index]
#        if label == 0:
        transform_params = get_params(self.opt, xray.size)
        xray_transform = get_transform(self.opt, transform_params, grayscale=True, run_type = self.run_type)
        xray = xray_transform(xray)
#        else:
#          transform_t = info[1]
#          transform_params = get_params(self.opt, xray.size)
#          xray_transform = get_transform_augumentation(self.opt, transform_params, transform_type = transform_t, grayscale=True, run_type = self.run_type)
#          xray = xray_transform(xray)

        # call segmentation here and return mask in output
        seg_out = self.segmentation_net(xray.to(self.device))
        seg_out = torch.argmax(seg_out.detach(), dim=1)

        xray = torch.cat((xray, xray, xray), 0)
        
        
        return {'A': xray, 'B': label, 'info': info, 'C': mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
      
    def get_label_distri(self):
        counts = np.array([0.,0.,0.,0.])
        for item in self.label_list:
          counts[item] += 1.
        counts = 1000./counts
        return torch.from_numpy(np.array([counts]))
          
