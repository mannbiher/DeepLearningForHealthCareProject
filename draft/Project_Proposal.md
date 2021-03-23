# Project Proposal

Up to 3 pages write-up + 1 page of references (minimum 6 papers)

## Guide

### Motivation

Why is this problem important? Why do we care this problem?

### Literature Survey

Conduct literature search to understand the state of arts and the
gap for solving the problem. Formulate the deep learning problem
in details (e.g., classification vs. predictive modeling vs.clustering
problem).

### Data

Describe the dataset you use, and elaborate on how you would play with
the data in your project.Preliminary results are encouraged but not
required. It is recommended to try to cover as many aspects as described
in project initiation to give you a better navigation in later period
of project phase. This is a crucial step, please do it on the first
day and never stops until the project.


Data:
5508 chest x-ray images
2874 independent patient cases

Two different dataset so need preprocessing to make them similar.

Very small dataset so lot of data processing is required.

### Approach

Identify the high-level technical approaches for the project (e.g., what
algorithms to use or pipelines to use). Identify clearly the success metric
that you would like to use (e.g., AUC, accuracy, recall, speedup in running
time).


It is a Classification problem

Use basis model
1. InceptionV3
2. Vgg19_bn
3. ResNeXt101
4. Resnet152
5. Densenet161

Ensemble model learning

Use Focal loss learning (FLANNEL Training algoritm)

1. Precision-Recall curve
2. ROC curve
3. Confusion Matrix
4. F1 score comparision accuracy of all methods

### Experimental Setup

Setup the analytic infrastructure for your project (including both hardware
and software environment, e.g., AWS or local clusters with Python,PyTorch and all
necessary packages).

### Timeline

Prepare a timeline and milestones of deliverables with reasonably
proposed task distributions for the entire project.
