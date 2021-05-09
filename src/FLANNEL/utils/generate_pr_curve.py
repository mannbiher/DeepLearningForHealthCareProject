import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from itertools import cycle

# Update results home value
results_home = 'results/'
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
def get_learner_y_score_y_true(learner):
    learner_y_score = np.empty((0, 4))
    learner_y_true = np.empty((0, 4))
    for ind in cv_index:
        if learner == 'ensemble':
            learner_results_dir = results_home + learner + 'Novel_20200719_gamma_10_multiclass_' + ind + '_focal'
            raw_df = pd.read_csv(learner_results_dir + '/result_detail.csv')
        else:
            learner_results_dir = results_home + learner + '_20200407_multiclass_' + ind
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


def generate_stats():
    stats = {}

    for base_learner in base_learners:
        learner_y_score, learner_y_true = get_learner_y_score_y_true(base_learner)
        stats[base_learner] = compute_learner_stats(learner_y_score, learner_y_true)
        print(base_learner + ' Average precision score, micro-averaged: {0:0.2f}'.format(stats[base_learner]["average_precision"]["micro"]))

    return stats

def plot_pr_curve(stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    for base_learner, color in zip(base_learners, colors):
        learner_stats = stats[base_learner]
        precision = learner_stats['precision']
        recall = learner_stats['recall']
        average_precision = learner_stats['average_precision']
        l, = plt.plot(recall["micro"], precision["micro"], color=color, lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.2f})'.format(base_learner, average_precision['micro']))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.35)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.60), prop=dict(size=14))
    plt.savefig(results_home + 'precision_recall_curve.png')
    plt.show()

def plot_roc_curve(stats):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold'])
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
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
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(results_home + "roc_curve.png")
    plt.show()

if __name__ == '__main__':
    stats = generate_stats()
    plot_pr_curve(stats)
    plot_roc_curve(stats)
