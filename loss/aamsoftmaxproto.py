#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.aamsoftmax as aamsoftmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.alpha = kwargs.get('alpha')
        self.aamsoftmax = aamsoftmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)
        params = torch.ones(2, requires_grad=True)
        self.params = torch.nn.Parameter(params)

        print('Initialised AAMSoftmaxPrototypical Loss with alpha ', self.alpha)

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        nlossS, prec1   = self.aamsoftmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(x.size()[1]))

        nlossP, _       = self.angleproto(x,None)

        nloss = nlossS + self.alpha * nlossP

        return nloss, prec1