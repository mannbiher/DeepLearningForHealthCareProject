import os
import glob

import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


from PIL import Image
import cv2

import header
import dataset
import model
import pickle

#from segmentation import header, dataset, model

def main():

    flag_save_JPG = True  # preprocessed, mask

    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        num_worker = header.num_worker
    else:
        device = torch.device("cpu") 
        num_worker = 0

    net = header.net

    # Load model
    model_dir = './data_preprocess/segmentation/model_v1.1.pth'
    if os.path.isfile(model_dir):
        print('\n>> Load model - %s' % (model_dir))
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['model_state_dict']) 
        test_sampler = checkpoint['test_sampler']
        print("  >>> Epoch : %d" % (checkpoint['epoch']))
        # print("  >>> JI Best : %.3f" % (checkpoint['ji_best']))
    else:
        print('[Err] Model does not exist in %s' % (model_dir))
        exit()


    # network to GPU
    net.to(device)

    dict_paths = [
        './data_preprocess/formal_covid_dict_ap.pkl',
        './data_preprocess/formal_kaggle_dict.pkl'
    ]

    for dict_path in dict_paths:
        formal_dict = pickle.load(open(dict_path, 'rb'))
        seg_dataset = dataset.SegmentationDataset(formal_dict)

        dataloader = DataLoader(seg_dataset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)

        with torch.no_grad():

            net.eval()

            for i, data in enumerate(dataloader, 0):

                outputs = net(data['input'].to(device))
                outputs = torch.argmax(outputs.detach(), dim=1)  

                outputs_max = torch.stack([dataset.one_hot(outputs[k], header.num_masks) for k in range(len(data['input']))])



                for k in range(len(data['input'])):

                    original_size, dir_case_id, dir_results = dataset.get_size_id(k, data['im_size'], data['ids'], header.net_label[1:])

                    post_output = [post_processing(outputs_max[k][j].numpy(), original_size) for j in range(1, header.num_masks)]

                    image_original = seg_dataset.get_original(i*header.num_batch_test+k)
                    img_id = data['ids'][k]
                    img_name = list(formal_dict[img_id]['image_dict'].keys())[0]
                    img_data = formal_dict[img_id]['image_dict'][img_name]
                    img_npy_path = img_data['path'] + '.image.npy'
                    img_msk_path = img_data['path'] + '.mask.npy'
                    np.save(img_npy_path, image_original)
                    np.save(img_msk_path, post_output[1] + post_output[2])
                    formal_dict[img_id]['image_dict'][img_name]['numpy_image_path'] = img_npy_path
                    formal_dict[img_id]['image_dict'][img_name]['numpy_mask_path'] = img_msk_path

                    if flag_save_JPG:
                        jpg_img = img_data['path'] + '.image.visualize.jpg'
                        jpg_msk = img_data['path'] + '.mask.visualize.jpg'
                        Image.fromarray(image_original.astype('uint8')).convert('L').save(jpg_img)
                        Image.fromarray(post_output[1]*255 + post_output[2]*255).convert('L').save(jpg_msk)

        pickle.dump(formal_dict, open(dict_path + '.segmented.pkl', 'wb'))


                    



def post_processing(raw_image, original_size, flag_pseudo=0):

    net_input_size = raw_image.shape
    raw_image = raw_image.astype('uint8')

    # resize
    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)
    else:
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)    

    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, net_input_size, interpolation=cv2.INTER_NEAREST)

    return raw_image

if __name__=='__main__':

    main()


