#!/bin/bash

## ===============================
## Download VoxCeleb dataset
## ===============================

# do it by yourself


## ===============================
## cut audio to fixed-length utts
## ===============================
python data_utils/cut_wav_to_fixLenSeg.py \
            /home/data/Speech_datasets/VoxCeleb1/dev \
            /home/data/Speech_datasets/VoxCeleb1/dev_seg

## ===============================
## make fbank feats
## ===============================            

# cd data_utils/kaldi/egs/Vox/v1,
# and use make_Vox_feats.sh to make fbank feats.