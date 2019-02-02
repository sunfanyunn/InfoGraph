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

from losses import *

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

# class Encoder(torch.nn.Module):
    # def __init__(self, num_features, dim, num_gc_layers):
        # super(Encoder, self).__init__()

        # # num_features = dataset.num_features
        # # dim = 32
        # self.num_gc_layers = num_gc_layers

        # # self.nns = []
        # self.convs = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList()

        # for i in range(num_gc_layers):

            # if i:
                # nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            # else:
                # nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            # conv = GINConv(nn)
            # bn = torch.nn.BatchNorm1d(dim)

            # self.convs.append(conv)
            # self.bns.append(bn)


    # def forward(self, x, edge_index, batch):
        # if x is None:
            # x = torch.ones((batch.shape[0], 1)).to(device)

        # xs = []
        # for i in range(self.num_gc_layers):

            # x = F.relu(self.convs[i](x, edge_index))
            # x = self.bns[i](x)
            # xs.append(x)
            # # if i == 2:
                # # feature_map = x2

        # xpool = [global_add_pool(x, batch) for x in xs]
        # x = torch.cat(xpool, 1)
        # return x, torch.cat(xs, 1)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

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
    def __init__(self, num_features, dim):
        super(Net, self).__init__()

        self.embedding_dim = dim
        self.local = False

        self.encoder = Encoder(num_features, dim)
        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

        self.local_d = FF(dim, dim)
        self.global_d = FF(2*dim, dim)
        
        self.local = True
        self.prior = False

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

    def unsup(self, data):

        y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure = 'JSD'
        if self.local:
            local_global_loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)
        else:
            local_global_loss = 0

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        
        return local_global_loss + PRIOR
