from model import GcnInfomax
from data_utils import read_graphfile
from graph_sampler import GraphSampler
import torch
from torch import optim
import os
import numpy as np
import gc
from evaluate_embedding import evaluate_embedding, draw_plot

#os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"

class Trainer:
    def __init__(self, args, real_train_graphs, train_graphs, test_graphs, extend):
        self.extend = extend
        self.batch_size = args.batch_size
        self.epoch_num = args.num_epochs
        self.neg_sampling_num = args.neg_sampling_num
        self.lr = args.lr
        self.args = args
        self.permutate = args.permutate

        print('number of real training graphs', len(real_train_graphs))
        print('number of training graphs', len(train_graphs))
        print('number of testing graphs', len(test_graphs))

        real_train_dataset_sampler = GraphSampler(real_train_graphs, 
                                        normalize=False,
                                        features=args.feature_type,
                                        no_node_labels=args.no_node_labels,
                                        no_node_attr=args.no_node_attr,
                                        max_num_nodes=args.max_num_nodes)

        args.max_num_nodes = real_train_dataset_sampler.max_num_nodes
        input_dim = real_train_dataset_sampler.input_dim
        node_label_dim = real_train_dataset_sampler.node_label_dim

        ###############
        #
        ###############
        train_dataset_sampler = GraphSampler(train_graphs, 
                                        normalize=False,
                                        features=args.feature_type,
                                        no_node_labels=args.no_node_labels,
                                        no_node_attr=args.no_node_attr,
                                        max_num_nodes=args.max_num_nodes,
                                         input_dim=input_dim,
                                         node_label_dim=node_label_dim)

        test_dataset_sampler = GraphSampler(test_graphs, 
                                        normalize=False,
                                        features=args.feature_type,
                                        no_node_labels=args.no_node_labels,
                                        no_node_attr=args.no_node_attr,
                                        max_num_nodes=args.max_num_nodes,
                                         input_dim=input_dim,
                                         node_label_dim=node_label_dim)


        print(len(real_train_dataset_sampler), len(train_dataset_sampler), len(test_dataset_sampler))
        self.real_train_dataloader = torch.utils.data.DataLoader(real_train_dataset_sampler,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset_sampler,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset_sampler,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        assert train_dataset_sampler.input_dim == test_dataset_sampler.input_dim \
               == real_train_dataset_sampler.input_dim

        args.input_dim = train_dataset_sampler.input_dim

        self.model = GcnInfomax(args)
        # self.num_graphs = len(self.dataloader)

    def evaluate(self):
        print('getting embeddings ...')
        train_embeddings, train_labels = self.model.get_embeddings(dataloader=self.train_dataloader)
        test_embeddings, test_labels = self.model.get_embeddings(dataloader=self.test_dataloader)
        return evaluate_embedding(train_embeddings, train_labels, test_embeddings, test_labels)

    def train(self):
        # torch.cuda.device(1)
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
        batch_num = 0

        history = {}
        history[-1] = (self.evaluate(), np.nan)
        print(history)
        accuracies = []
        for h in history.values():
            accuracies.append(h[0])
        print('=================')
        print('max', np.max(accuracies))
        print('mean', np.mean(accuracies))
        print('=================')

        for epoch in range(self.epoch_num+1):

            losses = []
            cur = 0
            for batch_idx, data in enumerate(self.real_train_dataloader):

                loss = self.model(data)
                optimizer.zero_grad()

                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                batch_num = batch_num + 1 
                # gc.collect()

            print('epoch %d, loss=%4.3f\n' %(epoch, np.mean(losses)))

            if epoch%self.args.log_interval == 0:
                history[epoch] = (self.evaluate(), np.mean(losses))
                # print(history)
                accuracies = []
                for h in history.values():
                    accuracies.append(h[0])
                print('=================')
                print('max', np.max(accuracies))
                print('mean', np.mean(accuracies))
                print('=================')

            
        fname = 'log'
        with open(fname, 'a+') as f:
            accuracies = []
            for h in history.values():
                accuracies.append(h[0])
            print('=================')
            print('max', np.max(accuracies))
            print('mean', np.mean(accuracies))
            print('=================')
            print("Optimization Finished!")

            tpe = ''
            if self.args.glob:
                tpe += 'global'
            if self.args.local:
                tpe += 'local'
            if self.args.prior:
                tpe += 'prior'
            if self.epoch_num == 100:
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                     self.args.DS,
                     'nor' if self.args.no_node_labels else 'all',
                     'concat' if self.args.concat else 'nonconcat',
                     self.args.neg_sampling_num,
                     self.args.lr,
                     tpe,
                     self.args.num_gc_layers,
                     self.args.loss_type,
                     history[-1][0],
                     history[0][0],
                     history[10][0],
                     history[20][0],
                     history[30][0],
                     history[40][0],
                     history[50][0],
                     history[60][0],
                     history[70][0],
                     history[80][0],
                     history[90][0],
                     history[self.epoch_num][0],
                     np.max(accuracies),
                     np.mean(accuracies),
                     'extend' if self.extend else 'no extend'))
            if self.epoch_num == 10:
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                     self.args.DS,
                     'nor' if self.args.no_node_labels else 'all',
                     'concat' if self.args.concat else 'noconcat',
                     self.args.neg_sampling_num,
                     self.args.lr,
                     tpe,
                     self.args.num_gc_layers,
                     self.args.loss_type,
                     history[-1][0],
                     history[0][0],
                     history[1][0],
                     history[2][0],
                     history[3][0],
                     history[4][0],
                     history[5][0],
                     history[7][0],
                     history[6][0],
                     history[8][0],
                     history[9][0],
                     history[self.epoch_num][0],
                     np.max(accuracies),
                     np.mean(accuracies),
                     'extend' if self.extend else 'no extend'))
