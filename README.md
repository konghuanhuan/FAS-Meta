# FAS-Meta

This is a face-anti-spoofing model.

## Install

pytorch==1.5.0

## Prepare

the face detector results in csv. if some photo not in csvfiles, that is to say there is no face in this photo, and when test, the no face photo scores 0.0.

ps: tainData:1 4 6 represent Scene，and it can be seen in traindata name.

## train

python trainFASMeta.py --imgroot your_image_root_path_before_(train) --results_path your_model_savepath

## test

python testFASMeta.py --imgroot your_image_root_path_before_(test val) --modelpath your_model_trainedpath --savepath results_save_path


## reference
https://github.com/rshaojimmy/AAAI2020-RFMetaFAS

