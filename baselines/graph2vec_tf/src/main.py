import argparse,os,logging,psutil,time
from joblib import Parallel,delayed

from utils import get_files
from train_utils import train_skipgram
from classify import perform_classification
from make_graph2vec_corpus import *
from time import time

logger = logging.getLogger()
logger.setLevel("INFO")


def main(args):
    '''
    :param args: arguments for
    1. training the skigram model for learning subgraph representations
    2. construct the deep WL kernel using the learnt subgraph representations
    3. performing graph classification using  the WL and deep WL kernel
    :return: None
    '''
    corpus_dir = args.corpus
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_size = args.embedding_size
    num_negsample = args.num_negsample
    learning_rate = args.learning_rate
    wlk_h = args.wlk_h
    label_filed_name = args.label_filed_name
    class_labels_fname = args.class_labels_file_name

    wl_extn = 'g2v'+str(wlk_h)
    assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
    # assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

    graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)
    logging.info('Loaded {} graph file names form {}'.format(len(graph_files),corpus_dir))


    t0 = time()
    wlk_relabel_and_dump_memory_version(graph_files, max_h=wlk_h, node_label_attr_name=label_filed_name)
    logging.info('dumped sg2vec sentences in {} sec.'.format(time() - t0))

    t0 = time()
    embedding_fname = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample,
                                     epochs, batch_size, output_dir, class_labels_fname)
    # logging.info('Trained the skipgram model in {} sec.'.format(round(time()-t0, 2)))

    # embedding_fname = '../embeddings/_dims_512_epochs_2_lr_0.5_embeddings.txt'
    # perform_classification (corpus_dir, wl_extn, embedding_fname, class_labels_fname)




def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("graph2vec")
    args.add_argument("-c","--corpus", default = "../data/kdd_datasets/ptc",
                      help="Path to directory containing graph files to be used for graph classification or clustering")

    args.add_argument('-l','--class_labels_file_name', default='../data/kdd_datasets/ptc.Labels',
                      help='File name containg the name of the sample and the class labels')

    args.add_argument('-o', "--output_dir", default = "../embeddings",
                      help="Path to directory for storing output embeddings")

    args.add_argument('-b',"--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument('-e',"--epochs", default=1000, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument('-d',"--embedding_size", default=1024, type=int,
                      help="Intended graph embedding size to be learnt")

    args.add_argument('-neg', "--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument('-lr', "--learning_rate", default=0.3, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--wlk_h", default=3, type=int, help="Height of WL kernel (i.e., degree of rooted subgraph "
                                                           "features to be considered for representation learning)")

    args.add_argument('-lf', '--label_filed_name', default='Label', help='Label field to be used '
                                                                         'for coloring nodes in graphs using WL kenrel')

    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
