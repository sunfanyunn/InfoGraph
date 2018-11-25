import numpy as np
import networkx as nx
from glob import glob
import os
import subprocess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def load_data(ds_name, use_node_labels):
    node2graph = {}
    Gs = []

    with open("./data/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "r") as f:
            c = 1
            for line in f:
                    node2graph[c] = int(line[:-1])
                    if not node2graph[c] == len(Gs):
                            Gs.append(nx.Graph())
                    Gs[-1].add_node(c)
                    c += 1

    with open("./data/%s/%s_A.txt"%(ds_name,ds_name), "r") as f:
            for line in f:
                    edge = line[:-1].split(",")
                    edge[1] = edge[1].replace(" ", "")
                    Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
            with open("./data/%s/%s_node_labels.txt"%(ds_name,ds_name), "r") as f:
                    c = 1
                    for line in f:
                            node_label = int(line[:-1])
                            Gs[node2graph[c]-1].node[c]['label'] = node_label
                            c += 1

            # for idx, g in enumerate(Gs):
                    # for n in g.nodes():
                            # _ = (g.node[n]['label'])

    labels = []
    with open("./data/%s/%s_graph_labels.txt"%(ds_name,ds_name), "r") as f:
            for line in f:
                    labels.append(int(line[:-1]))

    labels  = np.array(labels, dtype = np.float)
    return Gs, labels

def evaluate(DS, embeddings):
    graphs, labels = load_data(DS, False)

    # labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    kf = StratifiedKFold(n_splits=10, random_state=None)
    kf.shuffle=True
    accs=[];
    it = 0

    # print('Starting cross-validation')

    accuracies = []
    for train_index, test_index in kf.split(x, y):
        it += 1
        best_acc1 = 0

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
        classifier = SVC(C=10)
        # classifier = GridSearchCV(SVC(), params, cv=10, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    # print(accuracies)
    return np.mean(accuracies)

def preprocess(DS):
    Gs, labels = load_data(DS, False)
    print('number of graphs', len(Gs))
    try:
        os.mkdir(DS)
    except Exception as e:
        print(e)

    for i in range(len(Gs)):
        with open('{}/{}'.format(DS, i), 'w+') as f:
            for e in Gs[i].edges_iter():
                f.write('{} {}\n'.format(e[0], e[1]))
    print('done preprocessing')

if __name__ == '__main__':
    ds_name='MUTAG'
    print('classification')
    classification(ds_name, ds_name+'.vec')
    classification('ENZYMES', 'output')
