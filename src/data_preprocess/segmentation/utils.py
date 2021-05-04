import pickle
from PIL import Image
import numpy as np
import torch

import header

CLASSES = ['COVID-19', 'pneumonia_virus', 'pneumonia_bacteria', 'normal']

def save_datadict(out_data, path):
    data_dict = {}
    for (id_, class_, outfile, type_) in out_data:
        filename = outfile.rsplit('/',1)[-1]
        if id_ in data_dict:
#            class_var = [k for k, v in data_dict[id_]['class'].items()
 #                   if v==1][0]
 #           old_class = CLASSES.index(class_var)
            if filename not in data_dict[id_]['image_dict'].keys():
                data_dict[id_]['class'][class_] = 1
                data_dict[id_]['image_dict'][filename] = {
                        'path': outfile,
                        'type':type_}
            else:
                print(id_, class_, outfile,type_)
                print(data_dict[id_])
                raise ValueError(f'Duplicate entry in outdata {id_}')
        class_dict = {k: 0 for k in CLASSES}
        class_dict[CLASSES[class_]] = 1
        data_dict[id_] = {
            'image_dict': {filename: {'path': outfile,'type': type_ }},
            'class': class_dict
        }
    pickle.dump(data_dict, open(path, 'wb'))


def get_id_image(id_):
    id_parts = id_.split('_',2)
    image = id_parts[-1].rsplit('.',1)[0]
    original_id = '_'.join(id_parts[:-1])
    return original_id, image

def get_size_id(idx, size, case_id, classes):
    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    class_ = classes[idx]
    return original_size_w_h, case_id, class_

def apply_mask(original_image, mask_image):
    X_whole = Image.fromarray(original_image).resize(
        (header.post_resize, header.post_resize))
    X_whole = np.asarray(X_whole)
    X_whole_mask = Image.fromarray(mask_image).resize(
        (header.post_resize, header.post_resize))
    X_whole_mask = np.round(np.asarray(X_whole_mask))
    return np.multiply(X_whole, X_whole_mask)


def save_masked(original_image, mask_image, path):
    out = apply_mask(original_image, mask_image)
    np.savez_compressed(path, image=out)
