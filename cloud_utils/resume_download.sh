#!/usr/bin/env bash
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv5/model_best.pth.tar ../src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv5/model_best.pth.tar
aws s3 cp s3://alchemists-uiuc-dlh-spring2021-us-east-2/flannel/checkpoint/vgg19_bn_20200407_multiclass_cv5/log.txt ../src/explore_version_03/checkpoint/vgg19_bn_20200407_multiclass_cv5/log.txt
