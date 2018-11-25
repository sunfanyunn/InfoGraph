import networkx as nx
import numpy as np
from collections import defaultdict
import copy


def sp_kernel(g1, g2=None):
	if g2 != None:
		graphs = []
		for g in g1:
			graphs.append(g)
		for g in g2:
			graphs.append(g)
	else:
		graphs = g1

	N = len(graphs)
	all_paths = {}
	sp_counts = {}
	for i in range(N):
		sp_lengths = nx.shortest_path_length(graphs[i])
		sp_counts[i] = {}
		nodes = graphs[i].nodes()
		for v1 in nodes:
		    for v2 in nodes:
		        if v2 in sp_lengths[v1]:
		            label = tuple(sorted([graphs[i].node[v1]['label'], graphs[i].node[v2]['label']]) + [sp_lengths[v1][v2]])
		            if label in sp_counts[i]:
		                sp_counts[i][label] += 1
		            else:
		                sp_counts[i][label] = 1

		            if label not in all_paths:
		                all_paths[label] = len(all_paths)

	phi = np.zeros((N,len(all_paths)))

	for i in range(N):
		for label in sp_counts[i]:
		    phi[i,all_paths[label]] = sp_counts[i][label]

	if g2 != None:
		K = np.dot(phi[:len(g1),:],phi[len(g1):,:].T)
	else:
		K = np.dot(phi,phi.T)

	return K


def wl_kernel(g1, g2=None, h=6):
	if g2 != None:
		graphs = []
		for g in g1:
			graphs.append(g)
		for g in g2:
			graphs.append(g)
	else:
		graphs = g1
	
	labels = {}
	label_lookup = {}
	label_counter = 0

	N = len(graphs)

	orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

	# initial labeling
	ind = 0
	for G in graphs:
		labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
		node2index = {}
		for node in G.nodes():
		    node2index[node] = len(node2index)
		    
		for node in G.nodes():
		    label = G.node[node]['label']
		    if not (label in label_lookup):
		        label_lookup[label] = len(label_lookup)

		    labels[ind][node2index[node]] = label_lookup[label]
		    orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1
		
		ind += 1
		
	compressed_labels = copy.deepcopy(labels)

	# WL iterations
	for it in range(h):
		unique_labels_per_h = set()
		label_lookup = {}
		ind = 0
		for G in graphs:
		    node2index = {}
		    for node in G.nodes():
		        node2index[node] = len(node2index)
		        
		    for node in G.nodes():
		        node_label = tuple([labels[ind][node2index[node]]])
		        neighbors = G.neighbors(node)
		        if len(neighbors) > 0:
		            neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
		            node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
		        if not (node_label in label_lookup):
		            label_lookup[node_label] = len(label_lookup)
		            
		        compressed_labels[ind][node2index[node]] = label_lookup[node_label]
		        orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1
		        
		    ind +=1
		    
		labels = copy.deepcopy(compressed_labels)
	
	if g2 != None:
		K = np.zeros((len(g1), len(g2)))
		for it in range(-1, h):
			for i in range(len(g1)):
				for j in range(len(g2)):
				    common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][len(g1)+j].keys())
				    K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][len(g1)+j].get(k,0) for k in common_keys])
	else:
		K = np.zeros((N, N))
		for it in range(-1, h):
			for i in range(N):
				for j in range(N):
				    common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
				    K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][j].get(k,0) for k in common_keys])
				  	                            
	return K