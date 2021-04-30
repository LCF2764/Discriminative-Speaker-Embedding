## Discriminative Speaker Embedding

This repository contains the code and pre-trained models for our paper 'Learning Discriminative Speaker Embedding by Improving Aggregation Strategy and Loss Function for Speaker Verification'.

#### Data preparation

Use the `data_prepare.sh` script to prepare VoxCeleb1 and VoxCeleb2 dataset.

The Fbank features of the training input are made by  `data_utils/kaldi/egs/Vox/v1/make_Vox_feats.sh`

#### Training

```
python trainSpeakerNet.py \
--config config/exp_ResNetSE34L_vox1_trainSeg_MSA-NeXtVLAD_AAM+APL_MT234.yaml 
```

#### Evaluation

```
python trainSpeakerNet.py \
--config config/test_ResNetSE34L_vox1_trainSeg_MSA-NeXtVLAD_AAM+APL_MT234.yaml --eval
```



