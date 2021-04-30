#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e
root=/home/data_ssd/lcf/DSE_data
vox1_root=/home/data/Speech_datasets/VoxCeleb1
data=$root/data
exp=$root/exp

data_utils=../../../../../data_utils


stage=0

##############################################
# make fbank
##############################################
if [ $stage -le 0 ]; then

    local/make_SLR_data.pl $vox1_root/dev_seg $data/vox1_train
    local/make_voxceleb1_v2.pl $vox1_root dev $data/vox1_train_ori
    local/make_voxceleb1_v2.pl $vox1_root test $data/vox1_test

    for name in vox1_train vox1_train_ori vox1_test; do
        steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj 48 --cmd "$train_cmd" \
            $data/${name} $exp/make_fbank $root/fbank
        utils/fix_data_dir.sh $data/${name}

        sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
          $data/${name} $exp/make_fbank $root/fbank
        utils/fix_data_dir.sh $data/${name}        
    done
fi 

##############################################
# Generate ark-trials from voxceleb1_test.txt
##############################################
if [ $stage -le 1 ]; then
    python $data_utils/make_vox1test_trials.py $vox1_root/voxceleb1_test.txt \
                                               $data/vox1_test/feats.scp \
                                               $data/vox1_test/trials_ark
fi 

