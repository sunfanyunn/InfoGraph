import networkx as nx
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import LabelEncoder


class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, graphs, no_node_labels=False, no_node_attr=True, features='default', normalize=True, assign_feat='default', max_num_nodes=0, node_label_dim=None, input_dim=None):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.assign_feat_all = []

        self.no_node_labels = no_node_labels
        self.no_node_attr = no_node_attr

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in graphs])
        else:
            self.max_num_nodes = max_num_nodes

        print('max_num_nodes', self.max_num_nodes)
        self.G_list = graphs
        #if features == 'default':
        self.input_dim = 2
        # if 'feat' in graphs[0].node[0] and not self.no_node_attr:
            # print('Using node attributes')
            # self.feat_dim = graphs[0].node[0]['feat'].shape[0]
            # self.input_dim += self.feat_dim
        # else:
        self.feat_dim = 0
        print('No node attributes')


        if 'label' in graphs[0].node[0] and not self.no_node_labels:
            print('Using node labels')
            node_labels = []
            for idx, G in enumerate(graphs):
                for u in G.nodes():
                    node_labels.append(G.node[u]['label'])

            # tmp = np.array(tmp)
            # print(tmp)
            # tmp = np.argmax(tmp, axis=1)
            # node_labels = LabelEncoder().fit_transform(tmp)

            # cnt = 0
            # for idx, G in enumerate(self.G_list):
                # for u in G.nodes():
                    # G.node[u]['label'] = node_labels[cnt]
                    # cnt+=1
            self.node_label_dim = len(set(node_labels))
            # self.node_label_dim = graphs[0].node[0]
            self.input_dim += self.node_label_dim
        else:
            self.node_label_dim = 0
            print('No node labels')

        if input_dim is not None:
            self.input_dim = input_dim
        if node_label_dim is not None:
            self.node_label_dim = node_label_dim

        print('input dim', self.input_dim)
        print('node attribute dim', self.feat_dim)
        print('node label dim', self.node_label_dim)

        # self.feat_dim = self.input_dim#G_list[0].node[0]['feat'].shape[0]

        for G in graphs:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                # f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                # for i,u in enumerate(G.nodes()):
                    # f[i,:] = G.node[u]['feat']
                f = np.zeros((self.max_num_nodes, self.input_dim), dtype=float)
                if self.node_label_dim > 0:
                    for i,u in enumerate(G.nodes()):
                        if int(G.node[u]['label']) == 0:
                            f[i, int(G.node[u]['label'])] = 1.

                # if self.feat_dim > 0:
                    # for i,u in enumerate(G.nodes()):
                        # f[i, self.node_label_dim:-2] = G.node[u]['feat']

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
                # print(g_feat.shape)
                # print(f.shape)
                # input()
                self.feature_all.append(f)

            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = G.node[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in G.node[0]:
                    node_feats = np.array([G.node[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])
            
        # self.feat_dim = self.feature_all[0].shape[1]
        # self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
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

