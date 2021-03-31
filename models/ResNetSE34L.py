#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
# import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *
from models.VLAD import NeXtVLAD, NetVLAD

class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.use_final_bn = kwargs.get('final_bn')
        if self.use_final_bn:
            self.final_bn = nn.BatchNorm1d(nOut)

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
        
        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion

        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2

        elif self.encoder_type == "NeXtVLAD":
            get_dim = lambda e, c, d, g : (c*d*e)//g
            self.conv1_1 = nn.Conv2d(n_mels//8, 1, kernel_size=1)
            num_clusters_str, groups_str, expansion_str = kwargs.get('num_clusters'), kwargs.get('groups'), kwargs.get('expansion')# 8, 128, 8, 2
            vlad_drop = kwargs.get('vlad_drop')
            Cs = [int(i) for i in num_clusters_str.split('_')]
            Gs = [int(j) for j in groups_str.split('_')]
            Es = [int(k) for k in expansion_str.split('_')]
            self.NeXtVLAD = NeXtVLAD(num_clusters=Cs[-1], dim=num_filters[-1], alpha=100.0, groups=Gs[-1], 
                                     expansion=Es[-1], add_batchnorm=False, p_drop=vlad_drop)
            out_dim = get_dim(Es[-1],Cs[-1],num_filters[-1],Gs[-1])
            self.MultiStageAgg = kwargs.get('MultiStageAgg')
            if self.MultiStageAgg:
                self.MSA_type = kwargs.get('MSA_type') # 1234 234 14 24 34 124 134
                if self.MSA_type == '1234':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[0],Gs[0])+get_dim(Es[1],Cs[1],num_filters[1],Gs[1])+get_dim(Es[2],Cs[2],num_filters[2],Gs[2])
                    self.NeXtVLAD_L1 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[0], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                    self.NeXtVLAD_L2 = NeXtVLAD(num_clusters=Cs[1], dim=num_filters[1], alpha=100.0, groups=Gs[1], expansion=Es[1], p_drop=vlad_drop)
                    self.NeXtVLAD_L3 = NeXtVLAD(num_clusters=Cs[2], dim=num_filters[2], alpha=100.0, groups=Gs[2], expansion=Es[2], p_drop=vlad_drop)
                elif self.MSA_type == '234':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[1],Gs[0])+get_dim(Es[1],Cs[1],num_filters[2],Gs[1])
                    self.NeXtVLAD_L2 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[1], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                    self.NeXtVLAD_L3 = NeXtVLAD(num_clusters=Cs[1], dim=num_filters[2], alpha=100.0, groups=Gs[1], expansion=Es[1], p_drop=vlad_drop)
                elif self.MSA_type == '14':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[0],Gs[0])
                    self.NeXtVLAD_L1 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[0], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                elif self.MSA_type == '24':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[1],Gs[0])
                    self.NeXtVLAD_L2 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[1], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                elif self.MSA_type == '34':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[2],Gs[0])
                    self.NeXtVLAD_L3 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[2], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                elif self.MSA_type == '124':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[0],Gs[0])+get_dim(Es[1],Cs[1],num_filters[1],Gs[1])
                    self.NeXtVLAD_L1 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[0], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                    self.NeXtVLAD_L2 = NeXtVLAD(num_clusters=Cs[1], dim=num_filters[1], alpha=100.0, groups=Gs[1], expansion=Es[1], p_drop=vlad_drop)
                elif self.MSA_type == '134':
                    out_dim += get_dim(Es[0],Cs[0],num_filters[0],Gs[0])+get_dim(Es[1],Cs[1],num_filters[2],Gs[1])
                    self.NeXtVLAD_L1 = NeXtVLAD(num_clusters=Cs[0], dim=num_filters[0], alpha=100.0, groups=Gs[0], expansion=Es[0], p_drop=vlad_drop)
                    self.NeXtVLAD_L3 = NeXtVLAD(num_clusters=Cs[1], dim=num_filters[2], alpha=100.0, groups=Gs[1], expansion=Es[1], p_drop=vlad_drop)
                else:
                    pass
                

        elif self.encoder_type == "NetVLAD":
            num_clusters_str, ndim =  kwargs.get('num_clusters'), num_filters[-1]
            num_clusters = int(num_clusters_str)
            vlad_drop = kwargs.get('vlad_drop')
            self.conv1_1 = nn.Conv2d(n_mels//8, 1, kernel_size=1)
            self.NetVLAD = NetVLAD(num_clusters=num_clusters, dim=ndim, p_drop=vlad_drop)  
            out_dim = ndim * num_clusters
        else:
            out_dim = num_filters[3] * block.expansion * 2

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if len(x.size()) == 2:  x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        out5 = torch.mean(out4, dim=2, keepdim=True)
        
        if self.encoder_type == "SAP":
            x = out5.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = out5.permute(0,3,1,2).squeeze(-1)                   # x:[1, 135, 128]
            h = torch.tanh(self.sap_linear(x))                      # h:[1, 135, 128]
            w = torch.matmul(h, self.attention).squeeze(dim=2)      # w:[1, 135]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)   # w:[1, 135, 1]
            mu = torch.sum(x * w, dim=1)                            # mu:[1, 128]
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)
        elif self.encoder_type == "SP":
            x = out5.permute(0,3,1,2).squeeze(-1)
            mu = x.mean(dim=1, keepdim=True)
            var = torch.sum((x - mu)**2, dim=1, keepdim=True) / x.size()[1]
            std = torch.sqrt(var.clamp(min=1e-5))
            x = torch.cat((mu, std), dim=1)
        elif self.encoder_type == "NeXtVLAD":
            x = self.conv1_1(out4.permute(0, 2, 1, 3)).squeeze(1).permute(0,2,1)
            # x = out5.permute(0,3,1,2).squeeze(-1)
            x = self.NeXtVLAD(x)
            if self.MultiStageAgg: 
                if self.MSA_type == '1234': # 1234 234 14 24 34 124 134
                    x1 = torch.mean(out1, dim=2).permute(0, 2, 1)
                    x2 = torch.mean(out2, dim=2).permute(0, 2, 1)
                    x3 = torch.mean(out3, dim=2).permute(0, 2, 1)
                    l1 = self.NeXtVLAD_L1(x1)
                    l2 = self.NeXtVLAD_L2(x2)
                    l3 = self.NeXtVLAD_L3(x3)
                    x = torch.cat([x, l1, l2, l3], dim=1)
                elif self.MSA_type == '234':
                    x2 = torch.mean(out2, dim=2).permute(0, 2, 1)
                    x3 = torch.mean(out3, dim=2).permute(0, 2, 1)
                    l2 = self.NeXtVLAD_L2(x2)
                    l3 = self.NeXtVLAD_L3(x3)
                    x = torch.cat([x, l2, l3], dim=1)
                elif self.MSA_type == '14':
                    x1 = torch.mean(out1, dim=2).permute(0, 2, 1)
                    l1 = self.NeXtVLAD_L1(x1)
                    x = torch.cat([x, l1], dim=1)
                elif self.MSA_type == '24':
                    x2 = torch.mean(out2, dim=2).permute(0, 2, 1)
                    l2 = self.NeXtVLAD_L2(x2)
                    x = torch.cat([x, l2], dim=1)
                elif self.MSA_type == '34':
                    x3 = torch.mean(out3, dim=2).permute(0, 2, 1)
                    l3 = self.NeXtVLAD_L3(x3)
                    x = torch.cat([x, l3], dim=1)
                elif self.MSA_type == '124':
                    x1 = torch.mean(out1, dim=2).permute(0, 2, 1)
                    x2 = torch.mean(out2, dim=2).permute(0, 2, 1)
                    l1 = self.NeXtVLAD_L1(x1)
                    l2 = self.NeXtVLAD_L2(x2)
                    x = torch.cat([x, l1, l2], dim=1)
                elif self.MSA_type == '134':
                    x1 = torch.mean(out1, dim=2).permute(0, 2, 1)
                    x3 = torch.mean(out3, dim=2).permute(0, 2, 1)
                    l1 = self.NeXtVLAD_L1(x1)
                    l3 = self.NeXtVLAD_L3(x3)
                    x = torch.cat([x, l1, l3], dim=1)
        elif self.encoder_type == "NetVLAD":
            x = self.conv1_1(out4.permute(0, 2, 1, 3)).squeeze(1).permute(0,2,1)
            # x = out5.permute(0,3,1,2).squeeze(-1)
            x = self.NetVLAD(x)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        
        if self.use_final_bn:
            x = self.final_bn(x)

        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
