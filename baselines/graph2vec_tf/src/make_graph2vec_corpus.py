import os,sys,json,glob,copy,psutil,ast
from pprint import pprint
from time import time
import networkx as nx, numpy as np
from collections import defaultdict
from joblib import Parallel,delayed
from copy import deepcopy

label_to_compressed_label_map = {}

get_int_node_label = lambda l: int(l.split('+')[-1])

def initial_relabel(g,node_label_attr_name='Label'):
    global label_to_compressed_label_map
    # print(label_to_compressed_label_map)
    # raw_input()

    try:
        opfname = g+'.tmpg'
        g = nx.read_gexf(g)
    except:
        opfname = None
        pass

    nx.convert_node_labels_to_integers(g, first_label=0)  # this needs to be done for the initial interation only
    for node in g.nodes(): g.node[node]['relabel'] = {}

    for node in g.nodes():
        try:
            label = g.node[node][node_label_attr_name]
        except:
            # no node label referred in 'node_label_attr_name' is present, hence assigning an invalid compressd label
            g.node[node]['relabel'][0] = '0+0'
            continue

        if not label_to_compressed_label_map.has_key(label):
            compressed_label = len(label_to_compressed_label_map) + 1 #starts with 1 and incremented every time a new node label is seen
            label_to_compressed_label_map[label] = compressed_label #inster the new label to the label map
            g.node[node]['relabel'][0] = '0+' + str(compressed_label)
        else:
            g.node[node]['relabel'][0] = '0+' + str(label_to_compressed_label_map[label])

    if opfname:
        nx.write_gexf(g,opfname)
    else:
        return g

def wl_relabel(g, it):
    global label_to_compressed_label_map

    try:
        opfname = g+'.tmpg'
        g = nx.read_gexf(g+'.tmpg')
        new_g = deepcopy(g)
        for n in g.nodes():
            new_g.nodes[n]['relabel'] = ast.literal_eval(g.nodes[n]['relabel'])
        g = new_g
    except:
        opfname = None
        pass

    prev_iter = it - 1
    for node in g.nodes():
        prev_iter_node_label = get_int_node_label(g.nodes[node]['relabel'][prev_iter])
        node_label = [prev_iter_node_label]
        neighbors = list(nx.all_neighbors(g, node))
        neighborhood_label = sorted([get_int_node_label(g.nodes[nei]['relabel'][prev_iter]) for nei in neighbors])
        node_neighborhood_label = tuple(node_label + neighborhood_label)

        if not label_to_compressed_label_map.has_key(node_neighborhood_label):
            compressed_label = len(label_to_compressed_label_map) + 1
            label_to_compressed_label_map[node_neighborhood_label] = compressed_label
            g.node[node]['relabel'][it] = str(it) + '+' + str(compressed_label)
        else:
            g.node[node]['relabel'][it] = str(it) + '+' + str(label_to_compressed_label_map[node_neighborhood_label])

    if opfname:
        nx.write_gexf(g,opfname)
    else:
        return g

def dump_sg2vec_str (fname,max_h,g=None):
    if g is None:
        g = nx.read_gexf(fname+'.tmpg')
        new_g = deepcopy(g)
        for n in g.nodes():
            del new_g.nodes[n]['relabel']
            new_g.nodes[n]['relabel'] = ast.literal_eval(g.nodes[n]['relabel'])
        g = new_g

    opfname = fname + '.g2v' + str(max_h)

    if os.path.isfile(opfname):
        return

    with open(opfname,'w') as fh:
        for n,d in g.nodes(data=True):
            for i in xrange(0, max_h+1):
                try:
                    center = d['relabel'][i]
                except:
                    continue
                neis_labels_prev_deg = []
                neis_labels_next_deg = []

                if i != 0:
                    neis_labels_prev_deg = list(set([g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)]))
                    neis_labels_prev_deg.sort()

                NeisLabelsSameDeg = list(set([g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]))

                if i != max_h:
                    neis_labels_next_deg = list(set([g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)]))
                    neis_labels_next_deg.sort()

                nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = ' '.join (nei_list)

                sentence = center + ' ' + nei_list
                print>>fh, sentence

    if os.path.isfile(fname+'.tmpg'):
        os.system('rm '+fname+'.tmpg')


def wlk_relabel_and_dump_hdd_version(fnames,max_h,node_label_attr_name='Label'):
    global label_to_compressed_label_map

    for fname in fnames: initial_relabel(fname,node_label_attr_name)

    for it in xrange(1, max_h + 1):
        t0 = time()
        label_to_compressed_label_map = {}
        for fname in fnames: wl_relabel(fname,it)
        print 'WL iteration {} done in {} sec.'.format(it, round(time() - t0, 2))
        print 'num of WL rooted subgraphs in iter {} is {}'.format(it, len(label_to_compressed_label_map))

    t0 = time()
    for fname in fnames: dump_sg2vec_str(fname,max_h)
    print 'dumped sg2vec sentences in {} sec.'.format(round(time() - t0, 2))


def wlk_relabel_and_dump_memory_version(fnames,max_h,node_label_attr_name='Label'):
    global label_to_compressed_label_map

    t0 = time()
    graphs = [nx.read_gexf(fname) for fname in fnames]
    for g in graphs:
        if not g.is_directed() or not nx.is_directed_acyclic_graph(g):
            print(g)
            raw_input()
    print 'loaded all graphs in {} sec'.format(round(time() - t0, 2))

    t0 = time()
    graphs = [initial_relabel(g,node_label_attr_name) for g in graphs]
    print 'len(label_to_compressed_label_map)', len(label_to_compressed_label_map)
    print 'initial relabeling done in {} sec'.format(round(time() - t0, 2))

    for it in xrange(1, max_h + 1):
        t0 = time()
        label_to_compressed_label_map = {}
        graphs = [wl_relabel(g, it) for g in graphs]
        print 'WL iteration {} done in {} sec.'.format(it, round(time() - t0, 2))
        print 'num of WL rooted subgraphs in iter {} is {}'.format(it, len(label_to_compressed_label_map))

    t0 = time()
    for fname, g in zip(fnames, graphs):
        dump_sg2vec_str(fname, max_h, g)
    print 'dumped sg2vec sentences in {} sec.'.format(round(time() - t0, 2))


def main():

    ip_folder = '../data/kdd_datasets/ptc'
    max_h = 3

    all_files = sorted(glob.glob(os.path.join(ip_folder,'*gexf')))#[:100]
    print 'loaded {} files in total'.format(len(all_files))

    # wlk_relabel_and_dump_hdd_version(all_files,max_h)
    wlk_relabel_and_dump_memory_version(all_files,max_h)


if __name__ == '__main__':
    main()
