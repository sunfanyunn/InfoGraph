from data_utils import read_graphfile
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LogReg(nn.Module):
    # def __init__(self, ft_in, nb_classes):
        # super(LogReg, self).__init__()
        # self.fc = nn.Linear(ft_in, nb_classes)

        # for m in self.modules():
            # self.weights_init(m)

    # def weights_init(self, m):
        # if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight.data)
            # if m.bias is not None:
                # m.bias.data.fill_(0.0)

    # def forward(self, seq):
        # ret = self.fc(seq)
        # return ret

def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)

def evaluate_embedding(datadir, DS, embeddings, max_nodes=None):
    """
    context_file = open(os.path.join(datadir, DS, 'context'), 'r').readlines()
    embeddings = [np.sum([embeddings[int(idx),:] for idx in context_file[i].strip().split()], axis=0) for i in range(len(context_file))]
    # print(labels)
    """
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    kf = StratifiedKFold(n_splits=10, random_state=None)
    kf.shuffle=True
    accs=[];
    it = 0

    print('Starting cross-validation')
    """
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        it += 1
        best_acc1 = 0

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        search=True
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=10, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print('svc', np.mean(accuracies))

    accuracies = []
    for train_index, test_index in kf.split(x, y):
        it += 1
        best_acc1 = 0

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        search=True
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=10, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print('LinearSvc', np.mean(accuracies))
    """
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        it += 1
        best_acc1 = 0

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        search=True
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LogisticRegression(), params, cv=10, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print('logistic', np.mean(accuracies))
    return np.mean(accuracies)

if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
