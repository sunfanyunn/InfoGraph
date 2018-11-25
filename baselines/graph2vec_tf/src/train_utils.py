import os,logging
import numpy as np
from corpus_parser import Corpus
from utils import save_graph_embeddings
from skipgram import skipgram


def train_skipgram (corpus_dir, extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, output_dir, class_labels_fname):
    '''

    :param corpus_dir: folder containing WL kernel relabeled files. All the files in this folder will be relabled
    according to WL relabeling strategy and the format of each line in these folders shall be: <target> <context 1> <context 2>....
    :param extn: Extension of the WL relabled file
    :param learning_rate: learning rate for the skipgram model (will involve a linear decay)
    :param embedding_size: number of dimensions to be used for learning subgraph representations
    :param num_negsample: number of negative samples to be used by the skipgram model
    :param epochs: number of iterations the dataset is traversed by the skipgram model
    :param batch_size: size of each batch for the skipgram model
    :param output_dir: the folder where embedding file will be stored
    :return: name of the file that contains the subgraph embeddings (in word2vec format proposed by Mikolov et al (2013))
    '''

    op_fname = '_'.join([os.path.basename(corpus_dir), 'dims', str(embedding_size), 'epochs',
                         str(epochs),'lr',str(learning_rate),'embeddings.txt'])
    op_fname = os.path.join(output_dir, op_fname)
    # if os.path.isfile(op_fname):
        # logging.info('The embedding file: {} is already present, hence NOT training skipgram model '
                     # 'for subgraph vectors'.format(op_fname))
        # return op_fname

    logging.info("Initializing SKIPGRAM...")
    corpus = Corpus(corpus_dir, extn = extn, max_files=0)  # just load 'max_files' files from this folder
    corpus.scan_and_load_corpus()

    model_skipgram = skipgram(
        corpus_dir=corpus_dir,
        extn=extn,
        class_labels_fname=class_labels_fname,
        num_graphs=corpus.num_graphs,
        num_subgraphs=corpus.num_subgraphs,
        learning_rate=learning_rate,
        embedding_size=embedding_size,
        num_negsample=num_negsample,
        num_steps=epochs,  # no. of time the training set will be iterated through
        corpus=corpus,  # data set of (target,context) tuples
    )

    # final_embeddings = model_skipgram.train(corpus=corpus,batch_size=batch_size)
    model_skipgram.train(corpus=corpus,batch_size=batch_size)

    # logging.info('Write the matrix to a word2vec format file')
    # save_graph_embeddings(corpus, final_embeddings, op_fname)
    # logging.info('Completed writing the final embeddings, pls check file: {} for the same'.format(op_fname))
    # return op_fname

    # perform_classification(corpus_dir, extn, embeddings, class_labels_fname)



if __name__ == '__main__':
    pass
