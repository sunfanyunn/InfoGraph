import numpy as np
from tqdm import tqdm
from glob import glob
import collections
import numpy as np

def apk(actual, predicted, k=100000):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def dis(x, y, l=2):
    return np.linalg.norm(np.array(x)-np.array(y), l)

def convert_op_map(graph):
    new_op_prop_map = graph.new_vertex_property('int')
    op_prop_map = graph.vertex_properties['op']

    for n in graph.vertices():
        if op_prop_map[n] in op_enum:
            new_op_prop_map[n] = op_enum[op_prop_map[n]]
        else:
            op_enum[op_prop_map[n]] = max(op_enum.values()) + 1

    return new_op_prop_map

def evaluate_keyword(names, embeddings, keywords, kernel=False):
    keyword2idxs = collections.defaultdict(list)
    idxs = []
    for idx in range(len(names)):
        for keyword in keywords:
            if keyword in names[idx].lower():
                idxs.append((idx,keyword))
                keyword2idxs[keyword].append(idx)
    ret = []
    for idx in idxs:
        idx, keyword = idx[0], idx[1]

        if not kernel:
            q = embeddings[idx]

        rank = [i for i in range(len(names)) if i not in keyword2idxs[keyword]]
        if kernel:
            rank.sort(key= lambda x: -km[idx, x])
        else:
            rank.sort(key= lambda x: dis(q, embeddings[x]))
        # rank.sort(key= lambda x: -similarity[idx, x])
        
        actual = [x[0] for x in idxs]
        # actual.remove(idx)
        ret.append(apk(actual, rank))
    return np.mean(ret)

def evaluate_all(names, embeddings, kernel=False):
    # assert len(names) == len(embeddings)
    #keywords = ['conv', 'relu', 'adam', 'dropout', 'save', 'gradients']
    keywords = [['rnn','lstm', 'gru'], ['rmsprop', 'adam', 'sgd', 'optimizer/', 'adagrad', 'momentum', 'adadelta'], ['truncated_normal', 'random_uniform'], ['softmax_cross_entropy_loss', 'sigmoid_cross_entropy_loss']]
    for keyword in keywords:
        print('{},{}\n'.format(keyword, evaluate_keyword(names, embeddings, keyword, kernel=kernel)))


if __name__ == '__main__':
    DS = 'gitgraph-stars'
    op_enum = dict()
    op_enum['null'] = 0

    names = []
    fpath = './{}/data_graph_tool_subgraph/'.format(DS)
    num = len(glob(fpath + '*'))
    import json
    with open('./esc/gitgraph-stars-names.json', 'r') as f:
        names = json.load(f)

    graphs = []
    for i in tqdm(range(num)):
        g = load_graph('gitgraph-stars/data_graph_tool_subgraph/{}.gt'.format(i))
        graphs.append(g)


    embeddings = np.random.rand(num, 100)
    evaluate_all(names, embeddings)
    # get_co_matrix(names, embeddings=None)
    kernel=True
    if kernel:
        from kernel_method import get_kernel_matrix
        km = get_kernel_matrix(name='wl', num=len(names))
        evaluate_all(names, None, kernel=True)

    embeddings = []
    for i in tqdm(range(num)):
        cur = [0] * 286
        cnt = list(convert_op_map(graphs[i]))
        for c in cnt: cur[int(c)] += 1
        cur = np.array(cur)
        embeddings.append(cur)

    evaluate_all(names, embeddings)
    evaluate_all(names, [(cur/cur.sum() if cur.sum() else cur) for cur in embeddings])
