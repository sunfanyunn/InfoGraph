import zipfile
import collections
import numpy as np

import math
import random


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time

from train import Trainer
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph-skipgram Arguments.')
    # general arguments
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS',
            help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')
    parser.add_argument('--log-interval', dest='log_interval', type=int,
            help='logging interval (epoch)')
    # parser.add_argument('--load-on-train', dest='load_on_train', action='store_const',
            # const=True, default=False,
            # help='load graph data on training',

    # learning related arguments
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--neg-sampling-num', dest='neg_sampling_num', type=int,
            help='number of negative sampling')
    parser.add_argument('--permutate', dest='permutate', action='store_const', 
            const=True, default=False,
            help='Whether to permutate graph while training')

    # model related arguments
    parser.add_argument('--no-node-labels', dest='no_node_labels', action='store_const',
            const=True, default=False,
            help='whether to not use node labels or not')
    parser.add_argument('--no-node-attr', dest='no_node_attr', action='store_const',
            const=True, default=False,
            help='whether to not use node attributes or not')
    parser.add_argument('--max-num-nodes', dest='max_num_nodes', type=int,
            help='Maximum number of nodes (sample graghs with nodes exceeding the number.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    # parser.add_argument('--num-classes', dest='num_classes', type=int,
            # help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--loss-type', dest='loss_type',
            help='Loss function type')


    parser.set_defaults(datadir='data',
                        max_num_nodes=0,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=None,
                        batch_size=20,
                        num_epochs=100000,
                        # train_ratio=0.8,
                        # test_ratio=0.1,
                        log_interval=100,
                        num_workers=8,
                        # input_dim=18,
                        hidden_dim=50,
                        output_dim=100,
                        num_gc_layers=3,
                        dropout=0.0,
                        loss_type='bce',
                        method='base',
                        name_suffix='',
                        neg_sampling_num=1,
                        num_pool=1
                       )
    return parser.parse_args()

def main():
    args = arg_parse()
    # preprocess(args.datadir, args.DS, args.max_num_nodes)
    trainer = Trainer(args)
    # trainer.prepare_data()
    trainer.train()

if __name__ == '__main__':
    main()

