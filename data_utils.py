import collections
import numpy as np

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import math
import os
import random

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.encoders import GcnEncoderGraph

import json

import networkx as nx
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import LabelEncoder


class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, graphs, subgraphs,
            features='default', no_node_labels=False,
            no_node_attr=False, normalize=False, max_num_nodes=0):
        # self.adj_all = []
        # self.len_all = []
        # self.feature_all = []
        # self.label_all = []
        # self.assign_feat_all = []
        self.num_graphs = len(graphs)
        self.subgraph_label = [[] for _  in range(len(graphs))]
        for subgraphidx, subgraph in enumerate(subgraphs):
            self.subgraph_label[subgraph.graph['label']].append(self.num_graphs + subgraphidx)
        for i in range(len(graphs)):
            self.subgraph_label[i].append(i)

        self.local = (subgraphs is not None)
        self.G_list = graphs
        if self.local:
            self.G_list += subgraphs
        self.features = features
        self.no_node_labels = no_node_labels
        self.no_node_attr = no_node_attr
        self.normalize = normalize
        self.max_num_nodes = max_num_nodes

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in graphs])
        else:
            self.max_num_nodes = max_num_nodes

        print('max num nodes', self.max_num_nodes)

        self.input_dim = 2
        if 'feat' in graphs[0].node[0] and not self.no_node_attr:
            print('Using node attributes')
            self.feat_dim = graphs[0].node[0]['feat'].shape[0]
            self.input_dim += self.feat_dim
        else:
            self.feat_dim = 0
            print('No node attributes')

        
        if 'label' in graphs[0].node[0] and not self.no_node_labels:
            print('Using node labels')
            tmp = []
            for idx, G in enumerate(graphs):
                for u in G.nodes():
                    tmp.append(G.node[u]['label'])

            node_labels = LabelEncoder().fit_transform(tmp)
            cnt = 0
            for idx, G in enumerate(graphs):
                for u in G.nodes():
                    G.node[u]['label'] = node_labels[cnt]
                    cnt+=1

            self.node_label_dim = len(set(node_labels))
            self.input_dim += self.node_label_dim
        else:
            self.node_label_dim = 0
            print('No node labels')

        print('input dim', self.input_dim)
        print('node attribute dim', self.feat_dim)
        print('node label dim', self.node_label_dim)

        if self.max_num_nodes < 5000:
            self.load_on_train = False
            self.all_data = []
            print('loading graph data ...')
            for i in tqdm(range(len(graphs))):
                self.all_data.append(self.get_graph_data(i))
        else:
            self.load_on_train = True
            print('load on train')

    def get_graph_data(self, graphidx, processed=False, permutate=False):
        
        if processed and not permutate:
            return self.all_data[graphidx]
        # adj = self.adj_all[idx]

        G = self.G_list[graphidx]

        if permutate:
            tmp = np.arange(G.number_of_nodes())
            np.random.shuffle(tmp)
            tmp = list(tmp)
            mapping = {i:x for i,x in enumerate(tmp)}
            G = nx.relabel_nodes(G, mapping)


        adj = np.array(nx.to_numpy_matrix(G))
        if self.normalize:
            sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
            adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

        # self.adj_all.append(adj)
        # self.len_all.append(G.number_of_nodes())
        # self.label_all.append(G.graph['label'])

        ret_gfeat = None
        if self.features == 'default':
            f = np.zeros((self.max_num_nodes, self.input_dim), dtype=float)
            if self.node_label_dim > 0:
                for i,u in enumerate(G.nodes()):
                    f[i, int(G.node[u]['label'])] = 1.
            if self.feat_dim > 0:
                for i,u in enumerate(G.nodes()):
                    f[i, self.node_label_dim:-2] = G.node[u]['feat']

            degs = np.sum(np.array(adj), 1)
            degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()],
                                         'constant'),
                                  axis=1)
            clusterings = np.array(list(nx.clustering(G).values()))
            clusterings = np.expand_dims(np.pad(clusterings, 
                                                [0, self.max_num_nodes - G.number_of_nodes()],
                                                'constant'),
                                         axis=1)
            g_feat = np.hstack([degs, clusterings])
            f[:, -2:] = g_feat
            ret_gfeat=f
            
        elif self.features == 'id':
            # self.feature_all.append(np.identity(self.max_num_nodes))
            ret_gfeat = np.identity(self.max_num_nodes)

        elif self.features == 'deg':
            degs = np.sum(np.array(adj), 1)
            degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                  axis=1)
            # self.feature_all.append(degs)
            ret_gfeat = degs

        elif self.features == 'struct':
            assert False
            """
            degs = np.sum(np.array(adj), 1)
            degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()],
                                         'constant'),
                                  axis=1)
            clusterings = np.array(list(nx.clustering(G).values()))
            clusterings = np.expand_dims(np.pad(clusterings, 
                                                [0, self.max_num_nodes - G.number_of_nodes()],
                                                'constant'),
                                         axis=1)
            g_feat = np.hstack([degs, clusterings])
            if 'feat' in G.node[0]:
                node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                node_feats = np.array([G.node[i]['feat'] for i in range(G.number_of_nodes())])
                                    'constant')
                g_feat = np.hstack([g_feat, node_feats])

            # self.feature_all.append(g_feat)
            ret_gfeat = gfeat
            """

        # if self.assign_feat == 'id':
            # ret_assign_feat = np.hstack((np.identity(self.max_num_nodes), ret_gfeat))
        # else:
            # ret_assign_feat = ret_gfeat
            
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj': torch.from_numpy(np.array(adj_padded)),
                'feats': torch.from_numpy(np.array(ret_gfeat)),
                'num_nodes': torch.from_numpy(np.array(num_nodes))}
                # 'assign_feats': torch.from_numpy(np.array(ret_assign_feat))}
                # 'label': G.graph['label'],

        # self.feat_dim = self.feature_all[0].shape[1]
        # self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def get_batch(self, idxs, permutate=False):
        
        dim = len(idxs.shape)
        assert dim == 1
        # if idx.shape 
        ret_adj = []
        ret_feats = []
        ret_num_nodes = []
        ret_assign_feats = []

        ret_adj2 = []
        ret_feats2 = []
        ret_num_nodes2 = []
        ret_assign_feats2 = []
        
        for idx in idxs:

        #    idx = idx.item()
            data = self.get_graph_data(idx, processed=not self.load_on_train)
            ret_adj.append(data['adj'])
            ret_feats.append(data['feats'])
            ret_num_nodes.append(data['num_nodes'])
            # ret_assign_feats.append(data['assign_feats'])

            if self.local:
                subgraphidx = np.random.choice(self.subgraph_label[idx])
                idx = self.num_graphs + subgraphidx
                data = self.get_graph_data(idx, processed=not self.load_on_train, permutate=permutate)
                ret_adj2.append(data['adj'])
                ret_feats2.append(data['feats'])
                ret_num_nodes2.append(data['num_nodes'])
            else:
                data = self.get_graph_data(idx, processed=not self.load_on_train, permutate=permutate)
                ret_adj2.append(data['adj'])
                ret_feats2.append(data['feats'])
                ret_num_nodes2.append(data['num_nodes'])
                # ret_assign_feats2.append(data['assign_feats'])

        
        return {'adj': torch.from_numpy(np.stack(ret_adj)),
                'feats': torch.from_numpy(np.stack(ret_feats)),
                'num_nodes': torch.from_numpy(np.stack(ret_num_nodes))}, \
                {'adj': torch.from_numpy(np.stack(ret_adj2)),
                'feats': torch.from_numpy(np.stack(ret_feats2)),
                'num_nodes': torch.from_numpy(np.stack(ret_num_nodes2))}
                # 'assign_feats': torch.from_numpy(np.stack(ret_assign_feats2))

    def __len__(self):
        return len(self.G_list)

    # def __getitem__(self, idx):
        # return self.get_graph_data(idx)
        """
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats':self.assign_feat_all[idx].copy()}
        """

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line)]
        # node_labels = LabelEncoder().fit_transform(node_labels)
    except IOError:
        print('No node labels')

    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            if val == 0:
                label_has_zero = True
            graph_labels.append(val - 1)
    graph_labels = np.array(graph_labels)
    if label_has_zero:
        graph_labels += 1
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    # index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            # index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    # for k in index_graph.keys():
        # index_graph[k]=[u-1 for u in set(index_graph[k])]


    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        graphs.append(G)
      
        # add features and labels
    for nodeid, nl in enumerate(node_labels):
        nodeid += 1
        graphs[graph_indic[nodeid]-1].add_node(nodeid)
        # graphs[graph_indic[nodeid]-1][nodeid]['label'] = nl

    for idx, G in enumerate(graphs):
        # no graph labels needed
        G.graph['label'] = graph_labels[idx]
        for u in G.nodes():
            if len(node_labels) > 0:
                G.node[u]['label'] = node_labels[u-1]
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]

        graphs[idx] = G

    # relabeling
    for idx, G in enumerate(graphs):
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        G = nx.relabel_nodes(G, mapping)

        if max_nodes and G.number_of_nodes() > max_nodes:
            G = G.subgraph([i for i in range(0, max_nodes)])

        graphs[idx] = G

    # either max_nodes is 0 or None or 
    # return [g for g in graphs if not max_nodes or g.number_of_nodes() <= max_nodes]
    return graphs

def remove_singleton(graph):
    del_list = list()

    for v in graph.vertices():
        if (v.in_degree() + v.out_degree()) == 0:
            del_list.append(v)
    for v in reversed(sorted(del_list)):
        graph.remove_vertex(v)
    return graph
