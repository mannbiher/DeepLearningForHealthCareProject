import pickle
import json


def count_numbers(data_dict):
    covid_19, virus, bacteria, normal = 0, 0, 0, 0
    for key, value in data_dict.items():
        for image_name, info in value['image_dict'].items():
            if 'PA' in info['type'] or 'AP' in info['type']:
                if value['class']['COVID-19'] == 1:
                    covid_19 += 1
                if value['class']['pneumonia_virus'] == 1:
                    virus += 1
                if value['class']['pneumonia_bacteria'] == 1:
                    bacteria += 1
                if value['class']['normal'] == 1:
                    normal += 1

    return covid_19, virus, bacteria, normal


def test_index(formal_dict, index):
    og_id = f'{index}'
    img_name = list(formal_dict[og_id]['image_dict'].keys())[0]
    # print(img_name)
    class_dict = formal_dict[og_id]['class']
    class_dict_keys = list(class_dict.keys())
    class_dict_values = list(class_dict.values())
    class_index_list = list(filter(lambda i: i > 0, class_dict_values))
    # print(class_index_list)
    if len(class_index_list) > 0:
        value_index = class_index_list[0]
        class_name = class_dict_keys[class_dict_values.index(value_index)]
    else:
        class_name = 'Unknown'
        return -1


if __name__ == '__main__':
    with open('data_preprocess/formal_covid_dict_ap.pkl', 'rb') as f:
        c = pickle.load(f)

    with open('data_preprocess/formal_kaggle_dict.pkl', 'rb') as f:
        k = pickle.load(f)

    with open('data_preprocess/formal_kaggle_dict.pkl.segmented.pkl', 'rb') as f:
        k_sg = pickle.load(f)

    with open('data_preprocess/formal_covid_dict_ap.pkl.segmented.pkl', 'rb') as f:
        c_sg = pickle.load(f)

    print('new keys kaggle', set(k_sg.keys())-set(k.keys()))
    print('new keys covid', set(c_sg.keys())-set(c.keys()))
    print('missing keys kaggle', set(k.keys())-set(k_sg.keys()))
    print('missing keys covid', set(c.keys())-set(c_sg.keys()))

    print('kaggle', count_numbers(k))
    print('covid', count_numbers(c))
    print('kaggle segmented', count_numbers(k_sg))
    print('covid segmented', count_numbers(c_sg))
    # json.dump(k, open('kaggle.json','w'), sort_keys=True, indent=4)
    # json.dump(k_sg, open('kaggle_segmented.json','w'), sort_keys=True, indent=4)
    json.dump(c, open('covid.json', 'w'), sort_keys=True, indent=4)
    json.dump(c_sg, open('covid_segmented.json', 'w'),
              sort_keys=True, indent=4)

    # print(a)
"""
Total Counts
Kaggle:  0 1493 2780 1583
Covid: 479   16   42   20

AP, PA view new count
Kaggle AP: 0 1493 2780 1583
Kaggle PA: 0 0 0 0 0
Covid AP: 283 8 9 10 0
Covid PA: 196 8 33 10 0
"""
