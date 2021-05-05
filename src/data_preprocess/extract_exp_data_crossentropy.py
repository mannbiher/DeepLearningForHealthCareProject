import pickle
import random
import os
import csv
import argparse

import numpy as np

"""
Function that divides the data into K-folds (5 folds)
Each fold consisting of train, Valid and Test datasets
 with each of these datasets consisting of all 4 classes of images (Covid Virus, Non-Covid Pneumonia Virus, 
 Pneumonia Bacteria and Normal)
 
 The function prints the metrics and numbers of these 5 folds X 3 sets X 4 Classes of datasets
 
 Ex: 
    1-th detailed information of exp data
    1234 Train: Covid: 34 Pneumonia: 100 Pneumonia Bacteria: 400 Normal: 700
    312 Valid:  Covid: 12 Pneumonia: 50 Pneumonia Bacteria: 70 Normal 150
    345 Test:   Covid: 33 Pneumonia: 54 Pneumonia Bacteria: 80 Normal 178  
    
    2-th information of exp data
    ...
"""
def process_metadata(path):
    case_list = []
    count = 0
    covid_19, virus, bacteria, normal = 0, 0, 0, 0
    formal_covid_dict = pickle.load(open(path,'rb'))
    for key, value in formal_covid_dict.items():
        for image_name, info in value['image_dict'].items():
            if 'PA' in info['type'] or 'AP' in info['type']:
                if value['class']['COVID-19'] == 1:
                    case_list.append((info['path'], key+'_'+image_name, 0))
                    covid_19 += 1
                if value['class']['pneumonia_virus'] == 1:
                    case_list.append((info['path'], key+'_'+image_name, 1))
                    virus+=1
                if value['class']['pneumonia_bacteria'] == 1:
                    case_list.append((info['path'], key+'_'+image_name, 2))
                    bacteria += 1
                if value['class']['normal'] == 1:
                    case_list.append((info['path'], key+'_'+image_name, 3))
                    normal += 1
                count += 1
    print('All Images:', count) 
    print('Split:', len(case_list), covid_19, virus, bacteria, normal)
    return case_list


def shuffle_five_folds(case_list):
    random.shuffle(case_list)
    np = len(case_list)
    
    p1 = int(0.00*np)
    p2 = int(0.16*np)
    p3 = int(0.8*np)
    train_list_1 = case_list[:p1] + case_list[p2:p3]
    valid_list_1 = case_list[p1:p2]

    p1 = int(0.16*np)
    p2 = int(0.32*np)
    p3 = int(0.8*np)
    train_list_2 = case_list[:p1] + case_list[p2:p3]
    valid_list_2 = case_list[p1:p2]

    p1 = int(0.32*np)
    p2 = int(0.48*np)
    p3 = int(0.8*np)
    train_list_3 = case_list[:p1] + case_list[p2:p3]
    valid_list_3 = case_list[p1:p2]

    p1 = int(0.48*np)
    p2 = int(0.64*np)
    p3 = int(0.8*np)
    train_list_4 = case_list[:p1] + case_list[p2:p3]
    valid_list_4 = case_list[p1:p2]

    p1 = int(0.64*np)
    p2 = int(0.8*np)
    p3 = int(0.8*np)
    train_list_5 = case_list[:p1] + case_list[p2:p3]
    valid_list_5 = case_list[p1:p2]

    test_list = case_list[p2:]

    random.shuffle(train_list_1)
    random.shuffle(train_list_2)
    random.shuffle(train_list_3)
    random.shuffle(train_list_4)
    random.shuffle(train_list_5)
    random.shuffle(valid_list_1)
    random.shuffle(valid_list_2)
    random.shuffle(valid_list_3)
    random.shuffle(valid_list_4)
    random.shuffle(valid_list_5)
    random.shuffle(test_list)

    train_data = [train_list_1, train_list_2, train_list_3, train_list_4, train_list_5]
    valid_data = [valid_list_1, valid_list_2, valid_list_3, valid_list_4, valid_list_5]

    return train_data, valid_data, test_list



def write_five_folds(train_data, valid_data, test_list, out_dir):
    for index, (train_list, valid_list) in enumerate(zip(train_data,valid_data)):
        print ('%d-th detailed information of exp data'%(index+1))
        train_s = [0,0,0,0]
        test_s = [0,0,0,0]
        valid_s = [0,0,0,0]
        for x in train_list:
            train_s[x[2]] += 1
        for x in valid_list:
            valid_s[x[2]] += 1
        for x in test_list:
            test_s[x[2]] += 1
        print (train_s)
        print ('N of Train', len(train_list), 'covid:%d'%train_s[0], 'pneumonia_virus:%d'%train_s[1], 'pneumonia_bacteria:%d'%train_s[2], 'normal:%d'%train_s[3])
        print ('N of Valid', len(valid_list), 'covid:%d'%valid_s[0], 'pneumonia_virus:%d'%valid_s[1], 'pneumonia_bacteria:%d'%valid_s[2], 'normal:%d'%valid_s[3])
        print ('N of Test', len(test_list), 'covid:%d'%test_s[0], 'pneumonia_virus:%d'%test_s[1], 'pneumonia_bacteria:%d'%test_s[2], 'normal:%d'%test_s[3])

        with open(os.path.join(out_dir, f'data_statistic{index+1}.csv'),'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['N of Train', len(train_list), 'covid:%d'%train_s[0], 'pneumonia_virus:%d'%train_s[1], 'pneumonia_bacteria:%d'%train_s[2], 'normal:%d'%train_s[3]])
            csv_writer.writerow(['N of Valid', len(valid_list), 'covid:%d'%valid_s[0], 'pneumonia_virus:%d'%valid_s[1], 'pneumonia_bacteria:%d'%valid_s[2], 'normal:%d'%valid_s[3]])
            csv_writer.writerow(['N of Test', len(test_list), 'covid:%d'%test_s[0], 'pneumonia_virus:%d'%test_s[1], 'pneumonia_bacteria:%d'%test_s[2], 'normal:%d'%test_s[3]])

            train_path = os.path.join(out_dir, 'exp_train_list_cv%d.pkl'%(index+1))
            valid_path = os.path.join(out_dir, 'exp_valid_list_cv%d.pkl'%(index+1))
            test_path = os.path.join(out_dir, 'exp_test_list_cv%d.pkl'%(index+1))

            if os.path.exists(train_path):
                os.remove(train_path)

            if os.path.exists(valid_path):
                os.remove(valid_path)

            if os.path.exists(test_path):
                os.remove(test_path)

            pickle.dump(train_list, open(train_path,'wb'))
            pickle.dump(valid_list, open(valid_path,'wb'))
            pickle.dump(test_list, open(test_path,'wb'))



def setup_cli():
    parser = argparse.ArgumentParser(description='Create five folds for CV')
    
    parser.add_argument('--out-dir',type=str, help='output directory',required=True)
    parser.add_argument('--kaggle', type=str, help='Kaggle pickle file',required=True)
    parser.add_argument('--covid', type=str, help='Covid pickle file', required=True)
    # Datasets
    return parser.parse_args()
    

def main():
    args = setup_cli()
    print(args)
    case_list = []
    case_list += process_metadata(args.covid)
    case_list += process_metadata(args.kaggle)
    train, valid, test = shuffle_five_folds(case_list)
    try:
        os.mkdir(args.out_dir)
    except FileExistsError:
        print(args.out_dir, 'already exists. skipping creation.')
        pass
    write_five_folds(train, valid, test, args.out_dir)


if __name__ == '__main__':
    """Usage:
    
    python extract_exp_data_crossentropy \
    --out-dir ./data_preprocess/standard_data_multiclass_0922_crossentropy \
    --covid ./data_preprocess/formal_covid_dict_ap.pkl \
    --kaggle ./data_preprocess/formal_kaggle_dict.pkl
    """
    
    main()
    print ('finished')