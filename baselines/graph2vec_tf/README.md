# graph2vec

This repository contains the "tensorflow" implementation of our paper "graph2vec: Learning distributed representations of graphs". 
The paper could be found at: https://arxiv.org/pdf/1707.05005.pdf


#### Dependencies
This code is developed in python 2.7. It is ran and tested on Ubuntu 16.04.
It uses the following python packages:
1. tensorflow (version == 1.4.0)
2. networkx (version <= 2.0)
4. scikit-learn (+scipy, +numpy)

#####  The procedure for setting up graph2vec is as follows:
	1. git clone the repository (command: git clone https://github.com/MLDroid/graph2vec_tf.git )
	2. untar the data.tar.gz tarball

#####  The procedure for obtaining rooted graph vectors using graph2vec and performing graph classification is as follows:
	1. move to the folder "src" (command: cd src) (also make sure that kdd 2015 paper's (Deep Graph Kernels) datasets are available in '../data/kdd_datasets/dir_graphs/')
	2. run main.py --corpus <dataset of graph files> --class_labels_file_name <file containing class labels of graphs to be used for graph classification> file to:
		*Generate the weisfeiler-lehman kernel's rooted subgraphs from all the graphs 
		*Train skipgram model to learn graph embeddings. The same will be dumped in ../embeddings/ folder
		*Perform graph classification using the graph embeddings generated in the above step
	3. example: 
		*python main.py --corpus ../data/kdd_datasets/mutag --class_labels_file_name ../data/kdd_datasets/mutag.Labels 
		*python main.py --corpus ../data/kdd_datasets/proteins --class_labels_file_name ../data/kdd_datasets/proteins.Labels --batch_size 16 --embedding_size 128 --num_negsample 5
	

#### Other command line args:
	optional arguments:
		-h, --help            show this help message and exit
		-c CORPUS, --corpus CORPUS
				        Path to directory containing graph files to be used
				        for graph classification or clustering
		-l CLASS_LABELS_FILE_NAME, --class_labels_file_name CLASS_LABELS_FILE_NAME
				        File name containg the name of the sample and the
				        class labels
		-o OUTPUT_DIR, --output_dir OUTPUT_DIR
				        Path to directory for storing output embeddings
		-b BATCH_SIZE, --batch_size BATCH_SIZE
				        Number of samples per training batch
		-e EPOCHS, --epochs EPOCHS
				        Number of iterations the whole dataset of graphs is
				        traversed
		-d EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
				        Intended graph embedding size to be learnt
		-neg NUM_NEGSAMPLE, --num_negsample NUM_NEGSAMPLE
				        Number of negative samples to be used for training
		-lr LEARNING_RATE, --learning_rate LEARNING_RATE
				        Learning rate to optimize the loss function

		--wlk_h WLK_H         Height of WL kernel (i.e., degree of rooted subgraph
				        features to be considered for representation learning)
		-lf LABEL_FILED_NAME, --label_filed_name LABEL_FILED_NAME
				        Label field to be used for coloring nodes in graphs
				        using WL kenrel

## Contact ##
In case of queries, please email: annamala002@e.ntu.edu.sg OR XZHANG048@e.ntu.edu.sg

#### Reference 

	Please consider citing the follow paper when you use this code.
	@article{narayanangraph2vec,
	  title={graph2vec: Learning distributed representations of graphs},
	  author={Narayanan, Annamalai and Chandramohan, Mahinthan and Venkatesan, Rajasekar and Chen, Lihui and Liu, Yang}
	}





