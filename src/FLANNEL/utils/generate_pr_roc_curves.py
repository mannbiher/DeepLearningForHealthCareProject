import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from itertools import cycle

# Update results home value
result_sets = {
        'Patched FLANNEL': {
            'result_set': 'Patched FLANNEL',
            'results_home': 'patched_final_20210509/patched_results/results/',
            'base_learner_folder_format': '{0}_20200407_patched_{1}',
        },
        'Original FLANNEL': {
            'result_set': 'Original FLANNEL',
            'results_home': 'original_results/results/',
            'base_learner_folder_format': '{0}_20200407_multiclass_{1}'
        }
    }
base_learners = ["densenet161", "inception_v3", "resnet152", "resnext101_32x8d", "vgg19_bn", "ensemble"]
labels = ["Covid-19", "Pneumonia Virus", "Pneumonia Bacteria", "Normal"]
num_folds = 5


# create cv index for each learner
# depending on number of folds the index would be cv1, cv2, cv3 etc..
cv_index = []
for i in range(num_folds):
    fold = i + 1
    cv_index.append('cv' + str(fold))

def softmax_predictions(np_pred_cols):
    mx = np.max(np_pred_cols, axis=-1, keepdims=True)
    numerator = np.exp(np_pred_cols - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator/denominator


# function to return y pred and y true
def get_cv_y_score_y_true(df):
    pred_cols = df.iloc[:, :4]
    np_pred_cols = pred_cols.to_numpy()
    y_score = softmax_predictions(np_pred_cols)

    true_cols = df.iloc[:, -4:]
    y_true = np.zeros_like(true_cols.values)
    y_true[np.arange(len(true_cols)), true_cols.values.argmax(1)] = 1

    return y_score, y_true

# Wrapper function to read P & R, call F1 compute to create DF with for a base learner
def get_learner_y_score_y_true(result_set, learner):
    results_home = result_set['results_home']
    learner_y_score = np.empty((0, 4))
    learner_y_true = np.empty((0, 4))
    for ind in cv_index:
        if learner == 'ensemble':
            learner_results_dir = results_home + learner + 'Novel_20200719_gamma_10_multiclass_' + ind + '_focal'
            raw_df = pd.read_csv(learner_results_dir + '/result_detail.csv')
        else:
            learner_results_dir = results_home + result_set['base_learner_folder_format'].format(learner, ind)
            raw_df = pd.read_csv(learner_results_dir + '/result_detail_' + learner + '_test_' + ind + '.csv')
        y_score, y_true = get_cv_y_score_y_true(raw_df)
        learner_y_score = np.concatenate( (learner_y_score, y_score), axis=0)
        learner_y_true = np.concatenate( (learner_y_true, y_true), axis=0)

    return learner_y_score, learner_y_true

# generate precission, recall and roc auc stats for learner
def compute_learner_stats(y_score, y_true):
    precision = dict()
    recall = dict()
    average_precision = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true.ravel(), y_score.ravel(), average="micro")

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return {
            'precision': precision,
            'recall': recall,
            'average_precision': average_precision,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }


def generate_stats(result_set):
    stats = {}

    for base_learner in base_learners:
        learner_y_score, learner_y_true = get_learner_y_score_y_true(result_set, base_learner)
        stats[base_learner] = compute_learner_stats(learner_y_score, learner_y_true)
        print(result_set['result_set'] + ': ' + base_learner + ' Average precision score, micro-averaged: {0:0.2f}'.format(stats[base_learner]["average_precision"]["micro"]))

    return stats

def plot_result_set_pr_curve(result_set, stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    lines = []
    plt_labels = []
    for base_learner, color in zip(base_learners, colors):
        learner_stats = stats[base_learner]
        precision = learner_stats['precision']
        recall = learner_stats['recall']
        average_precision = learner_stats['average_precision']
        l, = plt.plot(recall["micro"], precision["micro"], color=color, lw=2)
        lines.append(l)
        plt_labels.append('{0} (area = {1:0.2f})'.format(base_learner, average_precision['micro']))

    fig = plt.gcf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(result_set['result_set'] + ' - Precision-Recall curve')
    plt.legend(lines, plt_labels, loc="lower left")
    file_prefix = re.sub(r'[^\sa-zA-Z0-9]', '', result_set['result_set']).lower().strip()
    plt.savefig(result_set['results_home'] + file_prefix + '_precision_recall_curve.png')
    plt.show()

def plot_result_set_covid_pr_curve(result_set, stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    covid_index = labels.index('Covid-19')
    lines = []
    plt_labels = []
    for base_learner, color in zip(base_learners, colors):
        learner_stats = stats[base_learner]
        precision = learner_stats['precision']
        recall = learner_stats['recall']
        average_precision = learner_stats['average_precision']
        l, = plt.plot(recall[covid_index], precision[covid_index], color=color, lw=2)
        lines.append(l)
        plt_labels.append('{0} (area = {1:0.2f})'.format(base_learner, average_precision[covid_index]))

    fig = plt.gcf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(result_set['result_set'] + ' - Covid 19 - Precision-Recall curve')
    plt.legend(lines, plt_labels, loc="lower left")
    file_prefix = re.sub(r'[^\sa-zA-Z0-9]', '', result_set['result_set']).lower().strip()
    plt.savefig(result_set['results_home'] + file_prefix + '_covid_19_precision_recall_curve.png')
    plt.show()

def plot_result_set_roc_curve(result_set, stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    lines = []
    plt_labels = []
    for base_learner, color in zip(base_learners, colors):
        learner_stats = stats[base_learner]
        tpr = learner_stats['tpr']
        fpr = learner_stats['fpr']
        roc_auc = learner_stats['roc_auc']
        plt.plot(
                fpr['micro'], tpr['micro'],
                label = '{0} (area = {1:0.2f})'.format(base_learner, roc_auc['micro']),
                color=color,
                linewidth=4
                )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(result_set['result_set'] + ' - ROC curve')
    plt.legend(loc="lower right")
    file_prefix = re.sub(r'[^\sa-zA-Z0-9]', '', result_set['result_set']).lower().strip()
    plt.savefig(result_set['results_home'] + file_prefix + "_roc_curve.png")
    plt.show()

def plot_result_set_covid_roc_curve(result_set, stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    covid_index = labels.index("Covid-19")
    for base_learner, color in zip(base_learners, colors):
        learner_stats = stats[base_learner]
        tpr = learner_stats['tpr']
        fpr = learner_stats['fpr']
        roc_auc = learner_stats['roc_auc']
        plt.plot(
                fpr[covid_index], tpr[covid_index],
                label = '{0} (area = {1:0.2f})'.format(base_learner, roc_auc[covid_index]),
                color=color,
                linewidth=4
                )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(result_set['result_set'] + ' - Covid 19 - ROC curve')
    plt.legend(loc="lower right")
    file_prefix = re.sub(r'[^\sa-zA-Z0-9]', '', result_set['result_set']).lower().strip()
    plt.savefig(result_set['results_home'] + file_prefix +  "_covid_19_roc_curve.png")
    plt.show()

def plot_ensemble_comparison_curves(result_stats):
    colors = cycle(['navy', 'darkorange'])
    covid_index = labels.index("Covid-19")
    plt.figure(figsize=(7, 8))
    for result_set, color in zip(result_stats.keys(), colors):
        ensemble_stats = result_stats[result_set]['ensemble']
        precision = ensemble_stats['precision']
        recall = ensemble_stats['recall']
        average_precision = ensemble_stats['average_precision']
        plt.plot(
                recall['micro'],
                precision['micro'],
                label='{0} ensemble (area = {1:0.2f})'.format(result_set, average_precision['micro']),
                color=color,
                lw=2
            )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Ensemle Comparison - Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig("ensemble_comparison_precision_recall_curve.png")
    plt.show()

    colors = cycle(['navy', 'darkorange'])
    plt.figure(figsize=(7, 8))
    for result_set, color in zip(result_stats.keys(), colors):
        ensemble_stats = result_stats[result_set]['ensemble']
        precision = ensemble_stats['precision']
        recall = ensemble_stats['recall']
        average_precision = ensemble_stats['average_precision']
        plt.plot(
                recall[covid_index],
                precision[covid_index],
                label='{0} ensemble (area = {1:0.2f})'.format(result_set, average_precision[covid_index]),
                color=color,
                lw=2
            )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Ensemle Covid-19 Comparison - Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig("ensemble_covid_19_comparison_precision_recall_curve.png")
    plt.show()

    plt.figure(figsize=(7, 8))
    for result_set, color in zip(result_stats.keys(), colors):
        ensemble_stats = result_stats[result_set]['ensemble']
        tpr = ensemble_stats['tpr']
        fpr = ensemble_stats['fpr']
        roc_auc = ensemble_stats['roc_auc']
        plt.plot(
                fpr['micro'],
                tpr['micro'],
                label='{0} ensemble (area = {1:0.2f})'.format(result_set, roc_auc['micro']),
                color=color,
                linewidth=4
                )
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble Comparison - ROC curve')
    plt.legend(loc="lower left")
    plt.savefig("ensemble_comparison_roc_curve.png")
    plt.show()

    plt.figure(figsize=(7, 8))
    for result_set, color in zip(result_stats.keys(), colors):
        ensemble_stats = result_stats[result_set]['ensemble']
        tpr = ensemble_stats['tpr']
        fpr = ensemble_stats['fpr']
        roc_auc = ensemble_stats['roc_auc']
        plt.plot(
                fpr[covid_index],
                tpr[covid_index],
                label='{0} ensemble (area = {1:0.2f})'.format(result_set, roc_auc[covid_index]),
                color=color,
                linewidth=4
                )
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble Covid-19 Comparison - ROC curve')
    plt.legend(loc="lower left")
    plt.savefig("ensemble_covid_19_comparison_roc_curve.png")
    plt.show()

if __name__ == '__main__':
    result_stats = {}
    for key,result_set in result_sets.items():
        stats = generate_stats(result_set)
        plot_result_set_pr_curve(result_set, stats)
        plot_result_set_roc_curve(result_set, stats)
        plot_result_set_covid_pr_curve(result_set, stats)
        plot_result_set_covid_roc_curve(result_set, stats)
        result_stats[key] = stats
    plot_ensemble_comparison_curves(result_stats)
