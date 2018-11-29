README for dataset NCI1


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


=== Description ===

NCI1 and NCI109 represent two balanced subsets of datasets of chemical compounds screened 
for activity against non-small cell lung cancer and ovarian cancer cell lines respectively
(Wale and Karypis (2006) and http://pubchem.ncbi.nlm.nih.gov). 


=== Previous Use of the Dataset ===

Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
Kernels from Propagated Information. Under review at MLJ.

Neumann, M., Patricia, N., Garnett, R., Kersting, K.: Efficient Graph Kernels by 
Randomization. In: P.A. Flach, T.D. Bie, N. Cristianini (eds.) ECML/PKDD, Notes in 
Computer Science, vol. 7523, pp. 378-393. Springer (2012).

Shervashidze, N., Schweitzer, P., van Leeuwen, E., Mehlhorn, K., Borgwardt, K.:
Weisfeiler-Lehman Graph Kernels. Journal of Machine Learning Research 12, 2539-2561 (2011)


=== References ===

N. Wale and G. Karypis. Comparison of descriptor spaces for chemical compound retrieval and 
classification. In Proc. of ICDM, pages 678–689, Hong Kong, 2006.

