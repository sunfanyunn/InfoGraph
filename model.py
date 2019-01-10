import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.encoders import *
import json
from torch import optim

class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.encoder = GcnEncoderGraph(input_dim=input_dim,
                                         hidden_dim=args.hidden_dim,
                                         num_layers=args.num_gc_layers,
                                         embedding_dim=args.embedding_dim,
                                         label_dim=None,
                                         bn=args.bn,
                                         concat=args.concat).cuda()

        self.l0 = nn.Linear(input_dim + args.embedding_dim, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class LocalDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        self.c2 = nn.Conv1d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        ret = self.c2(h)
        return self.c2(h)

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = args.embedding_dim
        if args.method == 'base':
            self.encoder = GcnEncoderGraph(input_dim=args.input_dim, 
                                             hidden_dim=args.hidden_dim,
                                             num_layers=args.num_gc_layers,
                                             embedding_dim=args.embedding_dim,
                                             label_dim=None,
                                             bn=args.bn,
                                             concat=args.concat).cuda()

        if args.concat:
            encoder_embedding_dim = self.embedding_dim + \
                                    (args.num_gc_layers-1) * args.hidden_dim
        else:
            encoder_embedding_dim = self.embedding_dim

        self.l0 = nn.Linear(encoder_embedding_dim, self.embedding_dim) 
        self.l1 = nn.Linear(self.embedding_dim, self.embedding_dim) 
        self.l2 = nn.Linear(self.embedding_dim, self.embedding_dim) 


    def forward(self, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        enc, features = self.encoder(h0, adj, batch_num_nodes)
        enc = self.l2(F.relu(self.l1(F.relu(self.l0(enc)))))
        return enc, features

class GcnInfomax(nn.Module):
  def __init__(self, args, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    # self.embedding_dim = args.output_dim
    # self.neg_sampling_size = args.neg_sampling_num

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.loss_type = args.loss_type
    self.embedding_dim = args.embedding_dim
    self.concat = args.concat
    self.local = args.local
    self.prior = args.prior
    self.glob = args.glob

    if self.concat:
        encoder_embedding_dim = self.embedding_dim + \
                                (args.num_gc_layers-1) * args.hidden_dim
    else:
        encoder_embedding_dim = self.embedding_dim

    self.encoder = Encoder(args)
    if self.local:
        self.local_d = LocalDiscriminator(encoder_embedding_dim + self.embedding_dim)
    if self.glob:
        self.global_d = GlobalDiscriminator(args, encoder_embedding_dim)
    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)
    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, data):

    batch_size = data['adj'].shape[0]
    # neg_sampling_size = self.neg_sampling_size

    y, M = self.encoder(data)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

    loss = 0.
    if self.glob:
        Ej = -F.softplus(-self.global_d(y, M, data)).mean()
        Em = F.softplus(self.global_d(y, M_prime, data)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        loss += GLOBAL

    if self.local:
        # y.shape: [batch_size, embedding_dim]
        # M.shape: [batch_size, max_num_nodes, encoder_embedding_dim]
        y_exp = y.unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, M.shape[1])

        # y_exp.shape [batch_size, 

        y_M = torch.cat((M.transpose(1, 2), y_exp), dim=1)
        y_M_prime = torch.cat((M_prime.transpose(1,2), y_exp), dim=1)
        

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta
        loss += LOCAL

    if self.prior:
        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        loss += PRIOR

    return loss


  def get_embeddings(self, dataloader):

      self.eval()
      assert not self.training

      labels = []
      embeddings = []
      with torch.no_grad():
          for batch_idx, data in enumerate(dataloader):
              emb, _ = self.encoder(data)
              embeddings.append(emb.cpu().numpy())
              labels.append(data['label'])

      embeddings = np.concatenate(embeddings, axis=0)
      self.train()
      assert self.training
      return embeddings, np.concatenate(labels, axis=0)

if __name__ == '__main__':
    print('getting batch ...')
    print(dataset_sampler.get_batch(torch.from_numpy(np.array([1]))))
    print()
