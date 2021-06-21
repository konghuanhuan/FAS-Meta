# FAS-Meta

This is a face-anti-spoofing model.

## Install

pytorch==1.5.0

## Overview

the face detector results in csv. if some photo not in csvfiles, that is to say there is no face in this photo, and when test, the no face photo scores 0.0.



## Data

train data: ./csv/1* ./csv/4/* ./csv/6/*

1 4 6 represent Scene，and it can be seen in traindata name. train dirname is Skin_Subject_Type_Scene_Light_Sensor,the 4rd is the split basic。for example，dirname 3_26_3_1_6_3 is labled in Scene 1. csv file contains 3 columns，path,box,label。path is image name(beigin from train,for example：train/3_81_1_1_1_3/0007.png)，box is face detect result（[xmin,ymin,xmax,ymax],for example："[215,322,908,1188]"），label is live face or not(0 is fake, 1 is real).

val data: ./csv/val/*
csv file contains 2 columns，path,box。path is image name(beigin from val,for example：val/0000/0001.png)，box is face detect result.

test data: ./csv/test/*
csv file contains 2 columns，path,box。path is image name(beigin from test,for example：test/0000/0001.png)，box is face detect result.


## train

python trainFASMeta.py --imgroot your_image_root_path_before_(train) --results_path your_model_savepath

## test

python testFASMeta.py --imgroot your_image_root_path_before_(test val) --modelpath your_model_trainedpath --savepath results_save_path


## reference
https://github.com/rshaojimmy/AAAI2020-RFMetaFAS

