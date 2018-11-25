import community
import networkx as nx
import os
from data_utils import read_graphfile
from tqdm import tqdm

def preprocess(datadir, DS, max_nodes=None):
    # log_path = os.path.join(datadir, DS, 'context')
    # graph_dir = os.path.join(datadir, DS, 'subgraphs')
    graph_dir = 'tmp/' + DS
    # if os.path.exists(log_path) and os.path.exists(subgraph_dir):
        # return

    try:
        os.mkdir(graph_dir)
    except:
        pass
    graphs = read_graphfile(datadir, DS, max_nodes)
    try:
        tmp = []
        for G in graphs:
            for u in G.nodes():
                tmp.append(G.node[u]['label'])
        print(set(tmp))
    except:
        pass

    print('number of nodes', len(tmp))
    print('number of graphs', len(graphs))
    print('max number of nodes', max([graph.number_of_nodes() for graph in graphs]))

    count = 0
    for graphid, G in tqdm(enumerate(graphs)):
        nx.write_gpickle(G, '{}/{}.gpickle'.format(graph_dir, graphid))
