#!/usr/bin/env bash
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv1/model_best.pth.tar ./src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv1/model_best.pth.tar
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv2/model_best.pth.tar ./src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv2/model_best.pth.tar
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv3/model_best.pth.tar ./src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv3/model_best.pth.tar
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv4/model_best.pth.tar ./src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv4/model_best.pth.tar
