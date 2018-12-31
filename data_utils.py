import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from glob import glob
import re

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
    # return [g for g in graphs if g.number_of_nodes() <= 2000]
    return graphs
