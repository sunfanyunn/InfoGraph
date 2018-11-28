from model import GraphSkipgram
from data_utils import get_data
import torch
from torch import optim
import os
import numpy as np

class Trainer:
    def __init__(self, args):
        self.embedding_dim = args.output_dim
        self.batch_size = args.batch_size
        self.epoch_num = args.num_epochs
        self.neg_sampling_num = args.neg_sampling_num
        self.lr = args.lr
        self.args = args
        self.permutate = args.permutate

        # with open(os.path.join(self.args.datadir, self.args.DS, 'context'), 'r') as f:
            # texts = f.readlines()
        # texts = [list(map(int, x[:-1].split())) for x in texts]
        # self.op = Options(texts)

        self.dataset_sampler = get_data(self.args.datadir, self.args.DS, self.args.max_num_nodes)
        args.input_dim = self.dataset_sampler.input_dim

        self.model = GraphSkipgram(args, self.dataset_sampler)
        # self.model.dataset_sampler = self.dataset_sampler
        self.num_graphs = len(self.dataset_sampler)
        print('nubmer of graphs', self.num_graphs)

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
        batch_num = 0


        history = {}
        for epoch in range(self.epoch_num):


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
                # print(pos_u['adj'].shape)
                # print(pos_u['feats'].shape)
                # print(pos_u['assign_feats'].shape)
                # input()
                # pos_v = self.dataset_sampler.get_batch(pos_v)
                neg_v, _ = self.dataset_sampler.get_batch(neg_v.flatten())

                optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v, self.batch_size)

                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)

                loss.backward()
                optimizer.step()

                losses.append(loss.data[0])
                if batch_num%10 == 0:
                    print('epoch %d, batch=%2d : loss=%4.3f\n' %(epoch, batch_num, loss.data[0]),end="")

                batch_num = batch_num + 1 

            if epoch%self.args.log_interval == 0:
                print('getting embeddings ...')
                embeddings = self.model.get_embeddings(total_num=len(self.dataset_sampler), batch_size=self.batch_size, permutate_sz=1)
                from evaluate_embedding import evaluate_embedding
                history[epoch] = (evaluate_embedding(self.args.datadir, self.args.DS, embeddings, self.args.max_num_nodes), np.mean(losses))
                print(history)



           # if epoch%100 == 0:
                # torch.save(self.model.state_dict(), './tmp/{}.epoch{}'.format(self.args.DS, epoch))
        print("Optimization Finished!")

