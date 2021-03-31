#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadFeats, test_dataset_loader
from tensorboardX import SummaryWriter
from datetime import datetime
from kaldiio import WriteHelper
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs);

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        self.nPerSpeaker = nPerSpeaker
        self.n_mels = kwargs.get('n_mels')

    def forward(self, data, label=None):
        data    = data.reshape(-1, self.n_mels, data.size()[-1]).cuda() 
        outp    = self.__S__.forward(data)
        if label == None:
            return outp
        else:
            outp    = outp.reshape(self.nPerSpeaker,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp,label)
            return nloss, prec1

class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, writer, **kwargs):

        self.__model__  = speaker_model

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        assert self.lr_step in ['epoch', 'iteration']
        self.scaler = GradScaler() 
        self.gpu = gpu
        self.mixedprec = mixedprec

        self.max_frames = kwargs.get('max_frames')
        self.writer = writer
        self.step = 0


    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    def train_network(self, loader, verbose):
        self.__model__.train()
        stepsize = loader.batch_size
        counter = 0
        index   = 0
        loss    = 0
        top1    = 0     # EER or accuracy
        tstart = time.time()
        time_count = 0
        
        for data, data_label in loader:
            data    = data.transpose(1,0)
            self.__model__.zero_grad()
            label   = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss    += nloss.detach().cpu()
            top1    += prec1.detach().cpu()
            counter += 1
            index   += stepsize

            telapsed = time.time() - tstart
            time_count += telapsed
            tstart = time.time()

            # tensorboard writer
            self.writer.add_scalar('loss', nloss.detach().cpu(), self.step)
            self.writer.add_scalar('Prec', float(prec1), self.step)
            self.writer.add_scalar('lr', self.__optimizer__.state_dict()['param_groups'][0]['lr'], self.step)
            self.step += 1

            if verbose:
                sys.stdout.write("\r(%d/%d) "%(index, loader.dataset.nFiles))
                sys.stdout.write("Loss %.4f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed))
                sys.stdout.write("- %.1fs"%(time_count))
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step(loss/counter)

        sys.stdout.write("\n")
        
        return (loss/counter, top1/counter)
        

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateFromList(self, test_list, nDataLoaderThread, print_interval=10, num_eval=5, **kwargs):
        
        self.__model__.eval()
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
        start_time  = time.time()
        time_count  = 0
        bs          = 0

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()
            for line in lines:
                gt, ark1, ark2 = line.strip().split()
                files.append(ark1)
                files.append(ark2)

        ## Get a list of unique file names
        # files = sum([x.strip().split()[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            num_frames = kwargs.get('eval_frames')
            if num_frames == 0:
                num_frames = data[0].size()[-1]
            
            batch_inps          = data[0].view(-1, kwargs.get('n_mels'), num_frames).cuda()
            embeds              = self.__model__(batch_inps).detach()
            embeds              = embeds.view(len(data[1]), num_eval, -1)
            for i, key in enumerate(data[1]):
                feats[key] = embeds[i]
            telapsed            = time.time() - tstart
            time_consume = time.time() - start_time
            time_count   += time_consume
            start_time   = time.time()
            bs           += len(data[1])
            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx*len(data[1]),len(setfiles),bs/telapsed,embeds.size()[-1]))
                sys.stdout.write(" - %.1fs"%(time_count))
        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for line in tqdm(lines, ncols=50):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__model__.module.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
            score = -1 * numpy.mean(dist)
            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

        return (all_scores, all_labels, all_trials);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Extract embeddings
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def extract_embeds(self, listfilename, print_interval=1, save_path='', num_eval=10, eval_frames=None):
        self.__model__.eval()
        arksscp     = {}
        files       = []
        tstart      = time.time()
        
        ## Read all lines
        with open(listfilename) as listfile:
            for line in listfile:
                data = line.strip().split()
                if len(data) == 3:
                    label, ark1, ark2 = data
                    files.append(ark1)
                    files.append(ark2)
                else:
                    utt, ark = data
                    files.append(ark)
                    arksscp[ark] = utt
        try:
            with open(listfilename.replace('trials_ark', 'feats.scp')) as featscp:
                for line in featscp:
                    utt, ark = line.strip().split()
                    arksscp[ark] = utt
        except:
            pass

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all embeddings to kaldi format file .ark
        if not os.path.exists(save_path): os.makedirs(save_path)
        with WriteHelper('ark,scp:{d}/embedding.ark,{d}/embedding.scp'.format(d=save_path)) as ark_writer:
            for file in tqdm(setfiles):

                inp1 = torch.FloatTensor(loadFeats(os.path.join(file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
                embedding = self.__model__(inp1).detach().cpu()
                if self.__model__.module.__L__.test_normalize:
                    embedding = F.normalize(embedding, p=2, dim=1)
                utt_name = arksscp[file]

                ark_writer(utt_name, embedding.squeeze().cpu().numpy())
                telapsed = time.time() - tstart

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

