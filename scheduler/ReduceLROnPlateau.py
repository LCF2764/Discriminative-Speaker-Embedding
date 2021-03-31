#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_decay, patience=5, 
													threshold=0.01, cooldown=5, verbose=True, min_lr=1e-6)

	lr_step = 'epoch'

	print('Initialised ReduceLROnPlateau LR scheduler')

	return sche_fn, lr_step
