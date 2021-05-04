import pickle
from mydataset import MyInferenceClass


def parse_data_dict(data_path):
    images = []
    ids = []
    classes = []
    types = []
    image_data_list = pickle.load(open(data_path, 'rb'))
    for key, value in image_data_list.items():
        for image_name, info in value['image_dict'].items():
            if 'PA' in info['type'] or 'AP' in info['type']:
                if value['class']['COVID-19'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(0)
                    types.append(info['type'])
                if value['class']['pneumonia_virus'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(1)
                    types.append(info['type'])
                if value['class']['pneumonia_bacteria'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(2)
                    types.append(info['type'])
                if value['class']['normal'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(3)
                    types.append(info['type'])
    return images, ids, classes, types
        

class SegmentationDataset(MyInferenceClass):
    def __init__(self, data_path):
        self.images, self.ids, self.classes, self.types = parse_data_dict(data_path)
        self.data_len = len(self.images)

    def __getitem__(self, index):
        old_data = super().__getitem__(index)
        old_data['classes'] = self.classes[index]
        old_data['types'] = self.types[index]
        return old_data
