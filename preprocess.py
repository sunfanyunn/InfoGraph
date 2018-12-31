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

def remove_singleton(graph):
    del_list = list()

    for v in graph.vertices():
        if (v.in_degree() + v.out_degree()) == 0:
            del_list.append(v)
    for v in reversed(sorted(del_list)):
        graph.remove_vertex(v)
    return graph

def make_subgraph_dataset(datadir, DS, method='louvain'):

    graphs = read_graphfile(datadir, DS)

    DS = DS + '-subgraph'
    try:
        os.mkdir('./data/'+DS)
    except:
        pass

    a_writer = open(f'./data/{DS}/{DS}_A.txt', 'w')
    graph_indicator_writer = open(f'./data/{DS}/{DS}_graph_indicator.txt', 'w')
    graph_labels_writer = open(f'./data/{DS}/{DS}_graph_labels.txt', 'w')
    node_labels_writer = open(f'./data/{DS}/{DS}_node_labels.txt', 'w')
    try:
        _ = graphs[0].node[0]['feat']
        feat = True 
    except:
        feat = False

    if feat:
        node_attributes_writer = open(f'./data/{DS}/{DS}_node_attributes.txt', 'w')

    node_cnt = 1
    graphid = 0
    subgraphid = 0
    for graph in tqdm(graphs):

        # graph = load_graph(f)
        # if graph.num_vertices() > 1000:
            # continue
        graphid += 1
        
        partition = community.best_partition(graph)
        for com in set(partition.values()):
            # count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]

        # for subgraph in corpus.get_subgraphs(graph):
            subgraph = graph.subgraph(list_nodes)

            subgraphid += 1
            for e in subgraph.edges():
                a_writer.write('{},{}\n'.format(node_cnt + e[0], node_cnt + e[1]))

            for v in subgraph.nodes():
                graph_indicator_writer.write('{}\n'.format(subgraphid))
                node_labels_writer.write('{}\n'.format(subgraph.node[v]['label']))
                if feat:
                    node_attributes_writer.write(', '.join(map(str, list(subgraph.node[v]['feat']))) + '\n')

            # graph_labels_writer.write('{}\n'.format(label2id[corpus.labels[graphid-1]]))
            graph_labels_writer.write('{}\n'.format(graphid))

        # node_cnt += subgraph.number_of_nodes()
        node_cnt += graph.number_of_nodes()

    a_writer.close()
    graph_indicator_writer.close()
    graph_labels_writer.close()
    node_labels_writer.close()
    if feat:
        node_attributes_writer.close()

if __name__ == '__main__':
    make_subgraph_dataset(datadir='./data/', DS='ENZYMES')
    make_subgraph_dataset(datadir='./data/', DS='PROTEINS_full')
    make_subgraph_dataset(datadir='./data/', DS='MUTAG')
    make_subgraph_dataset(datadir='./data/', DS='NCI1')
