# FLANNEL

## Update
TODO
fix below warnings
- /home/ubuntu/.virtualenvs/flannel/lib/python3.8/site-packages/torchvision/transforms/transforms.py:257: UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
  warnings.warn(
- FLANNEL/ensemble_step1.py:373: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  inputs, targets= torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
  

## Changes


### Ensemble Step1 volatile has no effect

```python
FLANNEL/ensemble_step2_ensemble_learning.py:375: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  inputs, targets= torch.autograd.Variable(inputs, volatile=True).float(), torch.autograd.Variable(targets)

```

```diff
-        inputs, targets= torch.autograd.Variable(inputs, volatile=True).float(), torch.autograd.Variable(targets)
+        inputs = inputs.float()
         # compute output
-        outputs = model(inputs)
-        loss = criterion(outputs, targets)
+        with torch.no_grad():
+            outputs = model(inputs)
+            loss = criterion(outputs, targets)
```

### Ensemble Step2 All tensors to be on same device

```python
Traceback (most recent call last):
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 420, in <module>
    main()
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 262, in main
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
  File "FLANNEL/ensemble_step2_ensemble_learning.py", line 315, in train
    loss = criterion(outputs, targets.type(torch.LongTensor).cuda())
  File "/home/neha/.virtualenvs/flannel/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/neha/DeepLearningForHealthCareProject/src/FLANNEL/models/proposedModels/loss.py", line 38, in forward
    loss = -1 * (1-p)**self.gamma * log_p * target_v * self.label_distri
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

```diff
         temp = torch.zeros(p.shape)
         if self.cuda_a:
           temp = temp.cuda()
+          if self.label_distri != None:
+            self.label_distri = self.label_distri.cuda()
         target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)

```




Views have been updated for Covid19 data
AP => AP and AP Erect

XRay data commit: 78543292f8b01d5e0ed1d0e15dce71949f0657bb

## Data Prepare
### Data Collect
1. Download CCX data: from https://github.com/ieee8023/covid-chestxray-dataset, put them into original_data/covid-chestxray-dataset-master
2. Download KCX data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, put them into original_data/chest_xray
### Data Preprocess
1. extract data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract data from KCX: data_preprocess/get_kaggle_data_dict.py
3. train segmentation _(follow steps provided in [segmentation readme](data_preprocess/segmentation/README.md))_
4. perform segmentation on images data_preprocess/segmentation/inference.py
4. reorganize CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy.py

## Model Training
### Base-modeler Learning
FLANNEL/ensemble_step1.py for 5 base-modeler learning [InceptionV3, Vgg19_bn, ResNeXt101, Resnet152, Densenet161]

(E.g. python ensemble_step1.py --arch InceptionV3)

### ensemble-model Learning
FLANNEL/ensemble_step2_ensemble_learning.py


## Codes for
@misc{ 
      
      title={FLANNEL: Focal Loss Based Neural Network Ensemble for COVID-19 Detection}, 
      
      author={Zhi Qiao and Austin Bae and Lucas M. Glass and Cao Xiao and Jimeng Sun}, 
      
      year={2020}, 
      
      eprint={2010.16039}, 
      
      archivePrefix={arXiv}, 
      
      primaryClass={eess.IV} 

}
