import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from infomax import *

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)

        out = self.set2set(out, data.batch)
        return out, feat_map[-1]



# class PriorDiscriminator(nn.Module):
    # def __init__(self, input_dim):
        # super().__init__()
        # self.l0 = nn.Linear(input_dim, input_dim)
        # self.l1 = nn.Linear(input_dim, input_dim)
        # self.l2 = nn.Linear(input_dim, 1)

    # def forward(self, x):
        # h = F.relu(self.l0(x))
        # h = F.relu(self.l1(h))
        # return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class Net(torch.nn.Module):
    def __init__(self, num_features, dim, use_unsup_loss=False, separate_encoder=False):
        super(Net, self).__init__()

        self.embedding_dim = dim
        self.separate_encoder = separate_encoder

        self.local = True
        # self.prior = False

        self.encoder = Encoder(num_features, dim)
        if separate_encoder:
            self.unsup_encoder = Encoder(num_features, dim)
            self.ff1 = FF(2*dim, dim)
            self.ff2 = FF(2*dim, dim)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

        if use_unsup_loss:
            self.local_d = FF(dim, dim)
            self.global_d = FF(2*dim, dim)
        
        self.init_emb()

    def init_emb(self):
      initrange = -1.5 / self.embedding_dim
      for m in self.modules():
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.fill_(0.0)


    def forward(self, data):
        out, M = self.encoder(data)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        pred = out.view(-1)
        return pred

    def unsup_loss(self, data):
        if self.separate_encoder:
            y, M = self.unsup_encoder(data)
        else:
            y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure = 'JSD'
        if self.local:
            loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)
        return loss


    def unsup_sup_loss(self, data):
        y, M =   self.encoder(data)
        y_, M_ = self.unsup_encoder(data)

        g_enc = self.ff1(y)
        g_enc1 = self.ff2(y_)

        measure = 'JSD'
        loss = global_global_loss_(g_enc, g_enc1, data.edge_index, data.batch, measure)

        return loss


    # def align_unsup_sup_loss(self, data):
        # y, M =   self.encoder(data)
        # y_, M_ = self.unsup_encoder(data)
        
        # return  F.mse_loss(y, y_)
