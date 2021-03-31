#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn 
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import kaldiio


def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def loadFeats(ark, max_frames, evalmode=True, num_eval=10, use_speed_perturb=False, use_feat_spec_aug=False):
    """
    args:
        ark: kaldi format ark
        max_frames: the number frames for training. when it is < 0, return whole length.
        evalmode: 
        num_eval: split the acoustic feats to num_eval parts.
    return:
        feats: Tensor shape of [dim, num_frames]
    """
    feat = kaldiio.load_mat(ark) # feat.shape = [num_frames, 40]
    feat = feat.T

    dim, num_frames = feat.shape
    if max_frames < 0:
        return torch.FloatTensor(feat)
        

    while num_frames <= max_frames:
        feat = numpy.hstack([feat, feat])
        num_frames = feat.shape[1]

    if evalmode:
        startframe = numpy.linspace(0, num_frames-max_frames, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(num_frames - max_frames))])

    feats = []
    if evalmode and max_frames==0:
        feats.append(feat)
    else:
        for asf in startframe:
            feats.append(feat[:, int(asf):int(asf)+max_frames])
    feat = numpy.stack(feats, axis=0)
    return torch.FloatTensor(feat)


class voxceleb_loader(Dataset):
    def __init__(self, dataset_file_name, max_frames):
        self.dataset_file_name = dataset_file_name;
        self.max_frames = max_frames
        
        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines();

        dictkeys = list(set([x.split()[0].split('-')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        self.label_dict = {}
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split();

            speaker_label = dictkeys[data[0].split('-')[0]];
            filename = data[1]

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = [];

            self.label_dict[speaker_label].append(lidx);
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

        self.nFiles = len(self.data_list)
    
    def __getitem__(self, indices):

        feat = []

        for index in indices:
            #audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            audio = loadFeats(self.data_list[index], self.max_frames, evalmode=False)
            feat.append(audio);

        feat = numpy.concatenate(feat, axis=0)
        feat = torch.FloatTensor(feat)
        return feat, self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_list  = test_list

    def __getitem__(self, index):
        #audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        feat = loadFeats(os.path.join(self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(feat), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class voxceleb_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):

        self.label_dict         = data_source.label_dict
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk;
        self.batch_size         = batch_size;
        
    def __iter__(self):
        
        dictkeys = list(self.label_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.label_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
           startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
           if flattened_label[ii] not in mixlabel[startbatch:]:
               mixlabel.append(flattened_label[ii])
               mixmap.append(ii)

        return iter([flattened_list[i] for i in mixmap])
    
    def __len__(self):
        return len(self.data_source)


def get_data_loader(dataset_file_name, batch_size, max_frames, max_seg_per_spk, nDataLoaderThread, nPerSpeaker, **kwargs):
    
    train_dataset = voxceleb_loader(dataset_file_name, max_frames)

    train_sampler = voxceleb_sampler(train_dataset, nPerSpeaker, max_seg_per_spk, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader


