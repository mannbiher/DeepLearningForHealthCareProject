#
# This utility is to generate the bar chart of FLANNEL vs Patched FLANNEL Covid-19 scores
# Make sure results_home and measure_detail* file names are adjusted accordingly while running this util.
#        results_home - line 12
#        measure_detail* files in create_f1df function
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Update results home value
results_home = '/home/ubuntu/DeepLearningForHealthCareProject/results/'
base_learners = ["densenet161", "inception_v3", "resnet152", "resnext101_32x8d", "vgg19_bn", "ensemble"]
types = ["Covid-19", "Pneumonia Virus", "Pneumonia Bacteria", "Normal"]

# create cv index for each learner
# depending on number of folds the index would be cv1, cv2, cv3 etc..
num_folds = 5
cv_index = []
for i in range(num_folds):
    fold = i + 1
    cv_index.append('cv' + str(fold))


# function to compute f1 score and macro-f1 score for each fold and each learner (for each file)
def compute_f1score(df):
    df.set_index('Type', inplace=True)
    micro_pr = df.iloc[0].mean()
    micro_re = df.iloc[1].mean()
    f1_df = pd.DataFrame(columns=types, index=['f1'])
    for x_type in types:
        prec = df[x_type]['Precision']
        recl = df[x_type]['Recall']
        f1_df[x_type]['f1'] = (2 * prec * recl) / (prec + recl)
    f1_df['Macro_F1_Score'] = f1_df.mean(axis=1)
    f1_df['Micro_F1_Score'] = (2 * micro_pr * micro_re) / (micro_pr + micro_re)
    return f1_df


# Wrapper function to read P & R, call F1 compute to create DF with for a base learner
def create_f1df(learner, dir):
    learner_f1 = pd.DataFrame(columns=types, index=cv_index)
    macro_f1_list = []
    micro_f1_list = []
    for ind in cv_index:
        result_home = results_home + dir + learner + '/' + ind
        # print(result_home)
        if learner == 'ensemble':
            raw_df = pd.read_csv(result_home + '/measure_detail.csv')
        else:
            raw_df = pd.read_csv(result_home + '/measure_detail_' + learner + '_test_' + ind + '.csv')
        f1_df = compute_f1score(raw_df)
        # print(f1_df['Macro_F1_Score']['f1'])
        for x_type in types:
            learner_f1[x_type][ind] = f1_df[x_type]['f1']
        macro_f1_list.append(f1_df['Macro_F1_Score']['f1'])
        micro_f1_list.append(f1_df['Micro_F1_Score']['f1'])
    learner_f1['Macro_F1_Score'] = macro_f1_list
    learner_f1['Micro_F1_Score'] = micro_f1_list
    return learner_f1


def addlabels(x,y,color='black'):
    for i in range(len(x)):
        plt.text(x[i], y[i], str(round(y[i], 4)), ha='right', va='bottom', fontsize='small', color=color)


def create_bar(scores, errors, micro_scores, micro_errors):
    #x_pos = [i for i,_ in enumerate(base_learners)]
    x_pos = [1, 4, 7, 10, 13, 16]
    x_micro_pos = [2, 5, 8, 11, 14, 17]
    colors = {'densenet161': 'papayawhip', 'inception_v3': 'blanchedalmond', 'vgg19_bn': 'bisque',
              'resnext101_32x8d': 'moccasin', 'resnet152': 'navajowhite', 'ensemble': 'limegreen'}
    colors_micro = {'densenet161': 'burlywood', 'inception_v3': 'tan', 'vgg19_bn': 'orange',
                    'resnext101_32x8d': 'orange', 'resnet152': 'goldenrod', 'ensemble': 'forestgreen'}
    fig = plt.figure()
    fig.add_axes([0.1, 0.1, 0.6, 0.75])
    plt.bar(x_pos, scores, color=colors.values(), yerr=errors, width=0.6, linewidth=0.1, figure=fig)
    plt.bar(x_micro_pos, micro_scores, color=colors_micro.values(), yerr=micro_errors, width=1.0, linewidth=0.1,
            figure=fig)
    addlabels(x_pos, scores)
    addlabels(x_micro_pos, micro_scores)
    plt.title("COVID-19 F1 score vs MICRO F1 Score", fontsize=10)
    plt.ylabel("F1 Scores")
    #plt.xlabel(base_learner)
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.xticks(x_pos, base_learners, rotation=23)
    plt.show()


def create_comp_bars(prev_scores, prev_errors, curr_scores, curr_errors):
    #x_pos = [i for i,_ in enumerate(base_learners)]
    bar_width = 8.0
    gap = (2 * bar_width) + 9.0

    prev_ind = 1
    curr_ind = prev_ind + bar_width + 0.8

    x_prev_pos = [prev_ind]
    x_curr_pos = [curr_ind]
    for i in range(len(base_learners) - 1):
        x_prev_pos.append(x_prev_pos[i] + gap)
        x_curr_pos.append(x_curr_pos[i] + gap)
    #x_prev_pos = [1, 4, 7, 10, 13, 16]
    #x_curr_pos = [2, 5, 8, 11, 14, 17]
    #print(x_prev_pos)
    #print(x_curr_pos)

    prev_colors = {'densenet161': 'papayawhip', 'inception_v3': 'blanchedalmond', 'vgg19_bn': 'bisque',
              'resnext101_32x8d': 'moccasin', 'resnet152': 'navajowhite', 'ensemble': 'limegreen'}
    curr_colors = {'densenet161': 'burlywood', 'inception_v3': 'tan', 'vgg19_bn': 'orange',
              'resnext101_32x8d': 'orange', 'resnet152': 'goldenrod', 'ensemble': 'forestgreen'}

    fig = plt.figure(figsize=(12,10), dpi=80)
    fig.add_axes([0.1, 0.1, 0.6, 0.75])

    plt.bar(x_prev_pos, prev_scores, color=prev_colors.values(), yerr=prev_errors, width=bar_width, linewidth=0.1, figure=fig)
    plt.bar(x_curr_pos, curr_scores, color=curr_colors.values(), yerr=curr_errors, width=bar_width, linewidth=0.1, figure=fig)

    addlabels(x_prev_pos, prev_scores, color='black')
    addlabels(x_curr_pos, curr_scores, color='blue')

    plt.title("Original FLANNEL vs Patched FLANNEL COVID-19 F1 scores", fontsize=15)
    plt.ylabel("F1 Scores")
    labels = list(prev_colors.keys())
    labels_curr = list(curr_colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=prev_colors[label]) for label in labels]
    handles_curr = [plt.Rectangle((0, 0), 1, 1, color=curr_colors[label]) for label in labels]
    #plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.5, 1))
    #plt.legend(handles_curr, labels_curr, loc='upper right', bbox_to_anchor=(1.5, 1))
    legend1 = plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.5, 1), title="Original FLANNEL Models")
    legend2 = plt.legend(handles_curr, labels_curr, loc='lower right', bbox_to_anchor=(1.5, 0.1), title="Patched FLANNEL Models")
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    plt.xticks(x_prev_pos, base_learners, rotation=23, fontsize=12)
    plt.show()


def get_f1_scores(dir):
    f1_scores = []
    sd_scores = []
    micro_f1_scores = []
    micro_sd_scores = []

    all_f1_scores = pd.DataFrame(columns=types, index=base_learners)
    all_sd_scores = pd.DataFrame(columns=types, index=base_learners)

    for base_learner in base_learners:
        learner_df = create_f1df(base_learner, dir)
        learner_sd = learner_df.std()
        learner_f1 = learner_df.mean()
        for x_type in types:
            all_f1_scores[x_type][base_learner] = learner_f1[x_type]
            all_sd_scores[x_type][base_learner] = learner_sd[x_type]
        f1_scores.append(learner_f1['Macro_F1_Score'])
        sd_scores.append(learner_sd['Macro_F1_Score'])
        micro_f1_scores.append(learner_f1['Micro_F1_Score'])
        micro_sd_scores.append(learner_sd['Micro_F1_Score'])
    all_f1_scores['Macro_F1_Score'] = f1_scores
    all_sd_scores['Macro_F1_Score'] = sd_scores
    all_f1_scores['Micro_F1_Score'] = micro_f1_scores
    all_sd_scores['Micro_F1_Score'] = micro_sd_scores
    scores = all_f1_scores['Covid-19'].tolist()
    errors = all_sd_scores['Covid-19'].tolist()
    print(all_f1_scores)
    print(all_sd_scores)
    return (scores, errors)


if __name__ == '__main__':
    prev_scores, prev_errors = get_f1_scores('old_results')
    curr_scores, curr_errors = get_f1_scores('final_results')
    #print(prev_scores)
    #print(curr_scores)
    create_comp_bars(prev_scores, prev_errors, curr_scores, curr_errors)