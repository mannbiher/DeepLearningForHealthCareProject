from mydataset import MyInferenceClass


def create_data_dict():
    pass



def parse_data_dict(data_path):
    images = []
    ids = []
    classes = []
    image_data_list = pickle.load(open(data_path, 'rb'))
    for key, value in image_data_list.items():
        for image_name, info in value['image_dict'].items():
            if 'PA' in info['type'] or 'AP' in info['type']:
                if value['class']['COVID-19'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(0)
                if value['class']['pneumonia_virus'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(1)
                if value['class']['pneumonia_bacteria'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(2)
                if value['class']['normal'] == 1:
                    images.append(info['path'])
                    ids.append(key+'_'+image_name)
                    classes.append(3)
    return images, ids, classes
        

class SegmentationDataset(MyInferenceClass):
    def __init__(self, data_path):
        self.images, self.ids, self.classes = parse_data_dict(data_path)
        self.data_len = len(self.images)

    def __getitem__(self, index):
        old_data = super().__getitem__(index)
        old_data['class'] = self.classes[index]



def get_size_id(idx, size, case_id, classes):

    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    class_ = classes[idx]

    return original_size_w_h, case_id, class_