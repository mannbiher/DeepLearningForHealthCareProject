import pickle

def test_index(formal_dict, index):
    og_id = f'{index}'
    img_name = list(formal_dict[og_id]['image_dict'].keys())[0]
    # print(img_name)
    class_dict = formal_dict[og_id]['class']
    class_dict_keys = list(class_dict.keys())
    class_dict_values = list(class_dict.values())
    class_index_list = list(filter(lambda i: i > 0, class_dict_values))
    # print(class_index_list)
    if len(class_index_list) > 0 :
        value_index = class_index_list[0]
        class_name = class_dict_keys[class_dict_values.index(value_index)]
    else:
        class_name = 'Unknown'
        return -1

if __name__ == '__main__':
    with open('data_preprocess/formal_covid_dict_ap.pkl','rb') as f:
    # with open('data_preprocess/formal_kaggle_dict.pkl','rb') as f:
        a= pickle.load(f)
    count = 0
    other_count = 0
    some_count = 0
    for i, val in a.items():
        if i=='101_':
            print(val)
        if test_index(a, i) == -1:
            count += 1
        else:
            other_count += 1

    print(count, other_count, some_count)
        
    
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