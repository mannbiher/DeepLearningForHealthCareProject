import pickle
import numpy as np
import random
import os
import csv

case_list = []
z0 = 0
formal_covid_dict = pickle.load(open('./data_preprocess/formal_covid_dict_ap.pkl','rb'))
for key, value in formal_covid_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'PA' in info['type'] or 'AP' in info['type']:
      if value['class']['COVID-19'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 0))
      if value['class']['pneumonia_virus'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 1))
      if value['class']['pneumonia_bacteria'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 2))
      if value['class']['normal'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 3))
      z0 += 1
print(len(case_list))

z1 = 0
formal_kaggle_dict = pickle.load(open('./data_preprocess/formal_kaggle_dict.pkl','rb'))
for key, value in formal_kaggle_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'PA' in info['type'] or 'AP' in info['type']:
      if value['class']['COVID-19'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 0))
      if value['class']['pneumonia_virus'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 1))
      if value['class']['pneumonia_bacteria'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 2))
      if value['class']['normal'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 3))
      z1 += 1
print (len(case_list))
print (z0, z1)
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
