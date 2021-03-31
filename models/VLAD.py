#! /usr/bin/python
# -*- encoding: utf-8 -*-
# https://github.com/ceshine/yt8m-2019/blob/95679eb3cf2ebc03c6c496319975cbe2dcb45af4/yt8m/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def general_weight_initialization(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        # print("Initing linear")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class TimeFirstBatchNorm1d(nn.Module):
    def __init__(self, dim, groups=None):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, tensor):
        _, length, dim = tensor.size()
        if self.groups:
            dim = dim // self.groups
        tensor = tensor.view(-1, dim)
        tensor = self.bn(tensor)
        if self.groups:
            return tensor.view(-1, length, self.groups, dim)
        else:
            return tensor.view(-1, length, dim)


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation
    Adapted from https://github.com/linrongc/youtube-8m/blob/master/nextvlad.py
    """

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 groups: int = 8, expansion: int = 2,
                 normalize_input=True, p_drop=0.25, add_batchnorm=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        assert dim % groups == 0, "`dim` must be divisible by `groups`"
        assert expansion > 1
        self.p_drop = p_drop
        self.cluster_dropout = nn.Dropout2d(p_drop)
        self.num_clusters = num_clusters
        self.dim = dim
        self.expansion = expansion
        self.grouped_dim = dim * expansion // groups
        self.groups = groups
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.add_batchnorm = add_batchnorm
        self.expansion_mapper = nn.Linear(dim, dim * expansion)
        # self.conv_reshape = nn.Conv2d(1, groups, (1, expansion*dim - self.grouped_dim + 1)).cuda()
        # self.conv1_1 = nn.Conv2d(groups, 1, kernel_size=1)

        if add_batchnorm:
            self.soft_assignment_mapper = nn.Sequential(
                nn.Linear(dim * expansion, num_clusters * groups, bias=False),
                TimeFirstBatchNorm1d(num_clusters, groups=groups)
            )
        else:
            self.soft_assignment_mapper = nn.Linear(
                dim * expansion, num_clusters * groups, bias=True)
        self.attention_mapper = nn.Linear(
            dim * expansion, groups
        )
        # (n_clusters, dim / group)
        self.centroids = nn.Parameter(
            torch.rand(num_clusters, self.grouped_dim))

        self.final_bn = nn.BatchNorm1d(num_clusters * self.grouped_dim)

        self._init_params()

    def _init_params(self):
        for component in (self.soft_assignment_mapper, self.attention_mapper,
                          self.expansion_mapper):
            for module in component.modules():
                general_weight_initialization(module)
        if self.add_batchnorm:
            self.soft_assignment_mapper[0].weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat((self.groups, self.groups))
            )
            nn.init.constant_(self.soft_assignment_mapper[1].bn.weight, 1)
            nn.init.constant_(self.soft_assignment_mapper[1].bn.bias, 0)
        else:
            self.soft_assignment_mapper.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat((self.groups, self.groups))
            )
            self.soft_assignment_mapper.bias = nn.Parameter(
                (- self.alpha * self.centroids.norm(dim=1)
                 ).repeat((self.groups,))
            )

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


    def forward(self, x, masks=None):
        """NeXtVlad Adaptive Pooling
        Arguments:
            x {torch.Tensor} -- shape: (n_batch, len, dim)
        Returns:
            torch.Tensor -- shape (n_batch, n_cluster * dim / groups)
        """
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # expansion
        # shape: (n_batch, len, dim * expansion)
        x = self.expansion_mapper(x)

        # soft-assignment
        # shape: (n_batch, len, n_cluster, groups)
        soft_assign = self.soft_assignment_mapper(x).view(x.size(0), x.size(1), self.num_clusters, self.groups)
        soft_assign = F.softmax(soft_assign, dim=2)

        # groups attention
        # shape: (n_batch, len, groups)
        attention = torch.sigmoid(self.attention_mapper(x))
        if masks is not None:
            attention = attention * masks[:, :, None]

        # (n_batch, len, n_cluster, groups, 1)
        activation = attention[:, :, None, :, None] * soft_assign[:, :, :, :, None]

        groups_x = x.view(x.size(0), x.size(1), 1, self.groups, self.grouped_dim)   # groups_x: [bs, T, 1, 8, 32]
        res = groups_x - self.centroids[:,None,:]                                   # res: [bs, T, 10, 8, 32]
        weight_res_groups = activation * res                                        # weight_res_groups: [bs, T, 10, 8, 32]
        weight_res = weight_res_groups.sum(-2)                                      # weight_res: [bs, T, 10, 32]

        vlad = weight_res.sum(1)
        V_dim = self.grouped_dim

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # flatten shape (n_batch, n_cluster * dim / groups)
        vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        vlad = self.final_bn(vlad)
        if self.p_drop:
            vlad = self.cluster_dropout(
                vlad.view(x.size(0), self.num_clusters, V_dim, 1)
            ).view(x.size(0), -1)
        return vlad


class NetVLAD(nn.Module):
    """NetVLAD layer implementation
       Adapted from https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    """

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True, p_drop=0.25, add_batchnorm=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.p_drop = p_drop
        self.cluster_dropout = nn.Dropout2d(p_drop)
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.add_batchnorm = add_batchnorm
        if add_batchnorm:
            self.soft_assignment_mapper = nn.Sequential(
                nn.Linear(dim, num_clusters, bias=False),
                TimeFirstBatchNorm1d(num_clusters)
            )
        else:
            self.soft_assignment_mapper = nn.Linear(dim, num_clusters, bias=True)
                

        # (n_clusters, dim)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        self.final_bn = nn.BatchNorm1d(num_clusters * dim)
        self._init_params()

    def _init_params(self):
        for module in self.soft_assignment_mapper.modules():
            general_weight_initialization(module)
        if self.add_batchnorm:
            self.soft_assignment_mapper[0].weight = nn.Parameter(2.0 * self.alpha * self.centroids)
            nn.init.constant_(self.soft_assignment_mapper[1].bn.weight, 1)
            nn.init.constant_(self.soft_assignment_mapper[1].bn.bias, 0)
        else:
            self.soft_assignment_mapper.weight = nn.Parameter(2.0 * self.alpha * self.centroids)
            self.soft_assignment_mapper.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


    def forward(self, x, masks=None):
        """NeXtVlad Adaptive Pooling
        Arguments:
            x {torch.Tensor} -- shape: (n_batch, len, dim)
        Returns:
            torch.Tensor -- shape (n_batch, n_cluster * dim)
        """
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # soft-assignment
        # shape: (n_batch, len, n_cluster)
        soft_assign = self.soft_assignment_mapper(x)
        soft_assign = F.softmax(soft_assign, dim=2)

        soft_assign_expand = soft_assign.unsqueeze(dim=-1) #(n_batch, len, n_cluster, 1)
        feat_broadcast = x.unsqueeze(dim=-2) # (n_batch, len, 1, dim)
        feat_res = feat_broadcast - self.centroids # (n_batch, len, n_cluster, dim)
        weight_res = soft_assign_expand * feat_res # (n_batch, len, n_cluster, dim)

        vlad = weight_res.sum(1)
        V_dim = self.dim
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = self.final_bn(vlad)
        if self.p_drop:
            vlad = self.cluster_dropout(
                vlad.view(x.size(0), self.num_clusters, V_dim, 1)
            ).view(x.size(0), -1)
        return vlad
