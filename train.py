from model import GraphSkipgram
from data_utils import read_graphfile, GraphSampler
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
            subgraphs = read_graphfile(args.datadir, args.DS + '-subgraphs', args.max_num_nodes)
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
                pos_u = []
                for i in range(self.batch_size):
                    pos_u.append((cur+i)%self.num_graphs)
                    cur += 1

                neg_v = []
                for i in range(len(pos_u)):
                    a = np.arange(self.num_graphs)
                    np.random.shuffle(a)
                    a = np.setdiff1d(a, pos_u[i])
                    neg_v.append(a[:self.neg_sampling_num])

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
                if batch_num%10 == 0:
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
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(self.args.DS,
                                                 'nor' if self.args.no_node_attr else 'all',
                                                 self.args.lr,
                                                 'local' if self.args.local else 'global',
                                                 self.args.num_gc_layers,
                                                 self.args.loss_type,
                                                 history[-1][0],
                                                 history[self.epoch_num][0],
                                                 np.max(accuracies),
                                                 np.mean(accuracies)))

