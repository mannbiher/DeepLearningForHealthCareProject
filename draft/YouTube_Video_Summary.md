Task:
Chest x-ray classification
Classes: COVID-19, Pneumonia virus, Pneumonia bacteria, Normal
Data: 5580 check x-ray images across 2874 patients
Method:
FLANNEL (Focal Loss based Neural Network Ensemble) model

1. Ensemble method (Integrate multiple base model together in combined model)
2. Focal loss (Handle class imbalance)

Result:
precion 0.7833 +- 0.07
recall  0.8609 +- 0.03
F1 score of 0.8168 +- 0.03

-------------------------------------------



Motivation
COVID-19 pandemic has ravaged the world on unprecedented scale. It has caused loss
of millions of lives and long lasting damages on surviving patients. X-ray imaging
is very important part of diagnosis of Covid-19 and other pneumonia and is often the
first-line diagnosis in many cases. Using deep learning for X-ray classification is
an ongoing reasearch area. As part of graduate course Deep learning for healthcare, we
have decided to reproduce and improve current research on COVID-19 classification using
X-ray.



our CS598 Deep learning for healthcare project, we have taken
First-line diagnosis of many diseases
can we differenciate COVID cases from other diseases based on chest x-ray images

Challenges
- Sample size
- Class imbalance fewer covid classes


Related Work
- Convolutional neural network models
Differenr CNN models
- Ensemble methods (Strategy for ensemble simple everaging , multi layer perceptron
training simple nueral network)
- Handling class imbalance (
  Traditional: repeat and duplicate rare classes, reduce examples majority classes)

- Design special loss to hanlde that (Focal loss)


- Two important datasets

Distribution of different classes

Data 

Two different datasets
Kaggle Chest X-ray images dataset (No covid classes)

    Covid Chest X-ray (CCX) dataset:
    This dataset containes COVID-19 pneumonia images as well few X-ray images from other classes.
    https://github.com/ieee8023/covid-chestxray-dataset
    
    Kaggle Chest X-ray (KCX) dataset:
    No covid classes
    This dataset contains normal, bacterial pneumonia, and nov-COVID-19 viral pneumonia.
    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia


AP view is common. PA view preprocessed to be similar to AP view.

Training/Test splits 80%/20% cross validation function 


Methods

1. Base model different CCN models with different architecture
2. 4 predications

---


Stage-2
Combine Ensemble Model Learning

Learn proper weight

--- Neural Weight module ---> outputs five different weights ---> 
Their prediction would be weigthed

compare 
Prediction
Real 

Focal loss

Neural Weight module

Concat ===> all predictions in long vector
==> 5*4 = 20
outer product over f => 20*20
===>
Flatten the matrix into 400 images
===>
Dense Neural Network
-===>
TanH
===>
Learner weights

Possible to assign negative weights as well

Pretrained model trained on ImageNet because 5000 is not sufficient

Standard Cross ENtropy loss

Equal weight of each class

FOcal Loss
LossFUnc = FocalLoss(

Heigh weight for poorly classified classes

1-ym ==> If y hat is close to 1
Downweight well-classified clases

Result: Base model performance

--
Average performance
Ensemble least amount of variation (Predictable and stable performance)




-------------------------------------------




