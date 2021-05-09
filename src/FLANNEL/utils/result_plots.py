import sys
import os
import csv

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

MODELS = ['inception_v3', 'resnext101_32x8d', 'resnet152',
          'densenet161', 'vgg19_bn', 'ensembleNovel']

FLANNEL_MODEL = 'ensembleNovel'
TYPES = ['Covid-19', 'Pneumonia Virus', 'Pneumonia Bacteria', 'Normal']


def read_measure(filename):
    n = len(TYPES)
    matrix = np.zeros((n, n), dtype=int)
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


def get_f1(filename):
    n = len(TYPES)
    y_true = []
    y_pred = []
    with open(filename) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # print(row)
            true_s = np.zeros(4).astype(float)
            pred_s = np.zeros(4).astype(float)
            pv = np.array(row[:4])
            rv = np.array(row[4:])
            p_id = np.argmax(pv)
            t_id = np.argmax(rv)
            true_s[t_id] += 1
            pred_s[p_id] += 1
            y_true.append(true_s)
            y_pred.append(pred_s)
    return f1_score(y_true, y_pred, average='macro')


def print_f1(result_dir):
    result_files = get_result_files(result_dir)
    # print(result_files)
    n = len(TYPES)
    for model, cv, filename in result_files:
        f1 = get_f1(filename)
        print(model, cv, f1)


def draw_cf(matrix, filename, cmap=None):
    df_cm = pd.DataFrame(matrix, index=TYPES,
                         columns=TYPES)
    print(df_cm)
    # plt.figure(figsize = (12,10))
    # sn.set(font_scale=1.1)
    fig = sn.heatmap(df_cm, annot=True, fmt="d",
                     center=800,
                     annot_kws={"size": 18},
                     cmap=cmap)
    fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def get_results(result_dir, cmap=None):
    result_files = get_result_files(result_dir)
    # print(result_files)
    n = len(TYPES)
    for model, cv, filename in result_files:
        matrix = read_measure(filename)
        filename = os.path.join(
            filename.rsplit('/', 1)[0],
            f'{model}_confusion_matrix_{cv}.png')
        draw_cf(matrix, filename, cmap)


def get_result_files(result_dir):
    return get_output_files(result_dir, 'result_detail.csv')


def get_measure_files(result_dir):
    return get_output_files(result_dir, 'measure_detail.csv')


def get_output_files(result_dir, pattern):
    output_files = []
    for (dirpath, dirnames, filenames) in os.walk(result_dir):
        if filenames and FLANNEL_MODEL in dirpath:
            for filename in filenames:
                if filename.endswith(pattern):
                    folder = dirpath.rsplit('/', 1)[-1]
                    model = folder.split('_', 1)[0]
                    cv = dirpath.rsplit('_', 2)[1]
                    output_files.append(
                        (model, cv, os.path.join(dirpath, filename)))
    return output_files


if __name__ == '__main__':
    # print_f1(sys.argv[1])
    # exit()
    if len(sys.argv) == 2:
        get_results(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[2] == 'togglecolor':
        get_results(sys.argv[1], cmap="YlGnBu")
    else:
        print('Usage')
        print('python result_plots.py result_dir [togglecolor]')
