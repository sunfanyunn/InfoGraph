from model import GraphSkipgram
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
        self.embedding_dim = args.output_dim
        self.batch_size = args.batch_size
        self.epoch_num = args.num_epochs
        self.neg_sampling_num = args.neg_sampling_num
        self.lr = args.lr
        self.args = args
        self.permutate = args.permutate

        print('reading graphfile...')
        graphs = read_graphfile(args.datadir, args.DS, args.max_num_nodes)
        print('number of graphs', len(graphs))
        if args.local:
            subgraphs = read_graphfile(args.datadir, args.local_ds, args.max_num_nodes)

            print('number of subgraphs', len(subgraphs))
        else:
            subgraphs = None
        self.dataset_sampler = GraphSampler(graphs, subgraphs,
                                            features=args.feature_type,
                                            no_node_labels=args.no_node_labels,
                                            no_node_attr=args.no_node_attr,
                                            max_num_nodes=args.max_num_nodes)

        args.input_dim = self.dataset_sampler.input_dim

        self.model = GraphSkipgram(args, self.dataset_sampler)
        self.num_graphs = len(self.dataset_sampler)
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
        embeddings = self.model.get_embeddings(total_num=len(self.dataset_sampler), batch_size=self.batch_size, permutate_sz=1)
        history[-1] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, self.args.max_num_nodes), np.nan)
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
            while cur < self.num_graphs:

                pos_u = [(cur+j)%self.num_graphs for j in range(self.batch_size)]
                cur += self.batch_size

                neg_v = []
                for i in range(self.batch_size):
                    a = np.arange(self.num_graphs)
                    np.random.shuffle(a)
                    a = np.setdiff1d(a, pos_u[i])
                    neg_v.append(a[:self.neg_sampling_num])

                assert len(pos_u) == len(neg_v)
                pos_u = np.array(pos_u)
                neg_v = np.array(neg_v)

                pos_u, pos_v = self.dataset_sampler.get_batch(pos_u, self.permutate)
                _ ,neg_v = self.dataset_sampler.get_batch(neg_v.flatten())

                optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v, self.batch_size)

                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                print('epoch %d, batch=%2d : loss=%4.3f\n' %(epoch, batch_num, loss.item()),end="")

                batch_num = batch_num + 1 
                # torch.cuda.empty_cache()
                # gc.collect()

            if epoch%self.args.log_interval == 0:
                print('getting embeddings ...')
                embeddings = self.model.get_embeddings(total_num=self.dataset_sampler.num_graphs, batch_size=self.batch_size, permutate_sz=1)
                history[epoch] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, self.args.max_num_nodes), np.mean(losses))
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
            if self.epoch_num == 100:
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                     self.args.DS,
                     'nor' if self.args.no_node_attr else 'all',
                     self.args.neg_sampling_num,
                     self.args.lr,
                     self.args.local_ds if self.args.local else 'global',
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
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                     self.args.DS,
                     'nor' if self.args.no_node_attr else 'all',
                     self.args.neg_sampling_num,
                     self.args.lr,
                     self.args.local_ds if self.args.local else 'global',
                     self.args.num_gc_layers,
                     self.args.loss_type,
                     history[-1][0],
                     history[0][0],
                     history[1][0],
                     history[2][0],
                     history[3][0],
                     history[4][0],
                     history[5][0],
                     history[6][0],
                     history[7][0],
                     history[8][0],
                     history[9][0],
                     history[self.epoch_num][0],
                     np.max(accuracies),
                     np.mean(accuracies)))
