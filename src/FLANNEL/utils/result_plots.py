import sys
import os
import csv

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODELS = ['inception_v3', 'resnext101_32x8d','resnet152',
    'densenet161','vgg19_bn','ensembleNovel']

FLANNEL_MODEL = 'ensembleNovel'
TYPES = ['Covid-19','Pneumonia Virus','Pneumonia Bacteria','Normal']

def read_measure(filename):
    n = len(TYPES)
    matrix = np.zeros((n,n),dtype=int)
    with open(filename) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # print(row)
            pv = np.array(row[:4])
            rv = np.array(row[4:])
            p_id = np.argmax(pv)
            t_id = np.argmax(rv)
            matrix[t_id, p_id] += 1
    return matrix


def draw_cf(matrix):
    df_cm = pd.DataFrame(matrix, index=TYPES,
                  columns = TYPES)
    print(df_cm)
    # plt.figure(figsize = (12,10))
    # sn.set(font_scale=1.1)
    fig = sn.heatmap(df_cm, annot=True, fmt="d",center=700, annot_kws={"size": 18})
    fig.set_yticklabels(fig.get_yticklabels(), rotation = 0)
    fig.set_xticklabels(fig.get_xticklabels(), rotation = 90)
    plt.savefig('confusion_matrix.png', bbox_inches = "tight")




def get_results(result_dir):
    measure_files = get_measure_files(result_dir)
    n = len(TYPES)
    matrix = np.zeros((n,n),dtype=int)
    for filename in measure_files:
        matrix += read_measure(filename)
    print(matrix/5)
    draw_cf(matrix//5)

    
    



def get_measure_files(result_dir):
    measure_files = []
    for (dirpath, dirnames, filenames) in os.walk(result_dir):
        if filenames and FLANNEL_MODEL in dirpath:
            for filename in filenames:
                if filename.endswith('result_detail.csv'):
                    measure_files.append(os.path.join(dirpath, filename))
    return measure_files
            



if __name__ == '__main__':
    if len(sys.argv) == 2:
        get_results(sys.argv[1])
        pass
    else:
        print('Usage')
        print('python result_plots.py result_dir')
