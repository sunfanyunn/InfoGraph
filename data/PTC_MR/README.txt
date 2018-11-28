README for dataset PTC_MR


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

The PTC dataset contains compounds labeled according to carcinogenicity 
on rodents divided into male mice (MM), male rats (MR), female mice (FM)
and female rats (FR).

The chemical data was obtained form http://www.predictive-toxicology.org/ptc/
and converted to graphs, where vertices represent atoms and edges 
represent chemical bonds. Explicit hydrogen atoms have been removed and
vertices are labeled by atom type and edges by bond type (single, double,
triple or aromatic). Chemical data was processed using the Chemistry 
Development Kit (v1.4).

Node labels:

  0  In
  1  P
  2  O
  3  N
  4  Na
  5  C
  6  Cl
  7  S
  8  Br
  9  F
  10  K
  11  Cu
  12  Zn
  13  I
  14  Ba
  15  Sn
  16  Pb
  17  Ca

Edge labels:

  0  triple
  1  double
  2  single
  3  aromatic


=== Previous Use of the Dataset ===

Kriege, N., Mutzel, P.: Subgraph matching kernels for attributed graphs. In: Proceedings
of the 29th International Conference on Machine Learning (ICML-2012) (2012).
