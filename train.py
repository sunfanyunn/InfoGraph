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
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epoch_num = args.num_epochs
        self.neg_sampling_num = args.neg_sampling_num
        self.lr = args.lr
        self.args = args
        self.permutate = args.permutate

        print('reading graphfile...')
        graphs = read_graphfile(args.datadir, args.DS, args.max_num_nodes)
        print('number of graphs', len(graphs))
        # if args.local:
            # subgraphs = read_graphfile(args.datadir, args.local_ds, args.max_num_nodes)

            # print('number of subgraphs', len(subgraphs))
        # else:
            # subgraphs = None
        dataset_sampler = GraphSampler(graphs, 
                                        normalize=False,
                                        features=args.feature_type,
                                        no_node_labels=args.no_node_labels,
                                        no_node_attr=args.no_node_attr,
                                        max_num_nodes=args.max_num_nodes)

        args.max_num_nodes = dataset_sampler.max_num_nodes
        self.dataloader = torch.utils.data.DataLoader(dataset_sampler,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        args.input_dim = dataset_sampler.input_dim

        self.model = GcnInfomax(args)
        self.num_graphs = len(self.dataloader)
        # assert self.num_graphs == len(graphs)
        # print('nubmer of graphs', self.num_graphs)

    def train(self):
        # torch.cuda.device(1)
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
        batch_num = 0


        history = {}

        print('getting embeddings ...')
        embeddings, labels = self.model.get_embeddings(dataloader=self.dataloader)
        history[-1] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, labels), np.nan)
        draw_plot(self.args.datadir, self.args.DS, embeddings, 'fig/{}_-1.png'.format(self.args.DS))
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
            for batch_idx, data in enumerate(self.dataloader):

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
                print('getting embeddings ...')
                embeddings, labels = self.model.get_embeddings(self.dataloader)
                history[epoch] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, labels), np.mean(losses))
                # history[batch_num] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, self.args.max_num_nodes), np.mean(losses))
                print(history)
                accuracies = []
                for h in history.values():
                    accuracies.append(h[0])
                draw_plot(self.args.datadir, self.args.DS, embeddings, 'fig/{}_{}.png'.format(self.args.DS, epoch))
                print('=================')
                print('max', np.max(accuracies))
                print('mean', np.mean(accuracies))
                print('=================')

            
           # if epoch%100 == 0:
                # torch.save(self.model.state_dict(), './tmp/{}.epoch{}'.format(self.args.DS, epoch))

        # fname = '{}-nor.log'.format(self.args.DS) if self.args.no_node_attr else '{}-all.log'.format(self.args.DS)
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
                     np.mean(accuracies)))
            if self.epoch_num == 10:
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
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
                     np.mean(accuracies)))
