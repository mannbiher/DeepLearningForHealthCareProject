import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Update results home value
results_home = '/Users/sreddyasi/CS598/pw/results/'
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
    f1_df = pd.DataFrame(columns=types, index=['f1'])
    for x_type in types:
        prec = df[x_type]['Precision']
        recl = df[x_type]['Recall']
        f1_df[x_type]['f1'] = (2 * prec * recl) / (prec + recl)
    f1_df['Macro_F1_Score'] = f1_df.mean(axis=1)
    return f1_df


# Wrapper function to read P & R, call F1 compute to create DF with for a base learner
def create_f1df(learner):
    learner_f1 = pd.DataFrame(columns=types, index=cv_index)
    macro_f1_list = []
    for ind in cv_index:
        result_home = results_home + learner + '/' + ind
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
    learner_f1['Macro_F1_Score'] = macro_f1_list
    return learner_f1


f1_scores = []
sd_scores = []

all_f1_scores = pd.DataFrame(columns=types, index=base_learners)
all_sd_scores = pd.DataFrame(columns=types, index=base_learners)

for base_learner in base_learners:
    learner_df = create_f1df(base_learner)
    learner_sd = learner_df.std()
    learner_f1 = learner_df.mean()
    # print(learner_f1)
    # f1_scores.append(learner_f1['Covid-19'])
    # sd_scores.append(learner_sd[0])
    for x_type in types:
        all_f1_scores[x_type][base_learner] = learner_f1[x_type]
        all_sd_scores[x_type][base_learner] = learner_sd[x_type]
    f1_scores.append(learner_f1['Macro_F1_Score'])
    sd_scores.append(learner_sd['Macro_F1_Score'])

all_f1_scores['Macro_F1_Score'] = f1_scores
all_sd_scores['Macro_F1_Score'] = sd_scores

scores = all_f1_scores['Covid-19'].tolist()
errors = all_sd_scores['Covid-19'].tolist()


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], str(round(y[i], 4)), ha='right', va='bottom', fontsize='small')


def create_bar(scores, errors):
    x_pos = [i for i,_ in enumerate(base_learners)]
    colors = {'densenet161': 'papayawhip', 'inception_v3': 'blanchedalmond', 'vgg19_bn': 'bisque',
              'resnext101_32x8d': 'moccasin', 'resnet152': 'navajowhite', 'ensemble': 'limegreen'}
    fig = plt.figure()
    fig.add_axes([0.1, 0.1, 0.6, 0.75])
    plt.bar(x_pos, scores, color=colors.values(), yerr=errors, width=0.6, linewidth=0.1, figure=fig)
    addlabels(x_pos, scores)
    plt.title("Illustration of COVID-19 F1 score vs rest - Error bars from 5-fold CV")
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.5, 1))
    #plt.xticks(x_pos, base_learners)
    plt.show()


create_bar(scores, errors)
