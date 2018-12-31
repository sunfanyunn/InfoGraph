import torch
import torch.utils.data
from tqdm import tqdm
import networkx as nx
import numpy as np
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
        self.local = (subgraphs is not None)
        if self.local:
            self.subgraph_label = [[] for _  in range(len(graphs))]
            for subgraphidx, subgraph in enumerate(subgraphs):
                self.subgraph_label[subgraph.graph['label']].append(self.num_graphs + subgraphidx)
            for i in range(len(graphs)):
                self.subgraph_label[i].append(i)

        self.G_list = graphs
        if self.local:
            self.G_list += subgraphs
            assert len(graphs) > self.num_graphs
            print(self.subgraph_label)
        self.features = features
        self.no_node_labels = no_node_labels
        self.no_node_attr = no_node_attr
        self.normalize = normalize
        self.max_num_nodes = max_num_nodes


        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in self.G_list])
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
            for idx, G in enumerate(self.G_list):
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
            for i in tqdm(range(len(self.G_list))):
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


            data = self.get_graph_data(idx, processed=not self.load_on_train)
            ret_adj.append(data['adj'])
            ret_feats.append(data['feats'])
            ret_num_nodes.append(data['num_nodes'])
            # ret_assign_feats.append(data['assign_feats'])

            if self.local: 
                idx = np.random.choice(self.subgraph_label[idx])

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
        return self.num_graphs
