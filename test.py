import torch
from sklearn.metrics import accuracy_score
from data_utils import read_graphfile
import numpy as np
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from glob import glob

def evaluate_embedding(datadir, DS, embeddings):
    graphs = read_graphfile(datadir, DS, max_nodes=None)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
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
        # classifier = SVC(C=10)
        classifier = GridSearchCV(SVC(), params, cv=10, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    # print(accuracies)
    return np.mean(accuracies)

if __name__ == '__main__':
    from main import arg_parse
    from train import Trainer
    from model import GraphSkipgram
    args = arg_parse()
    datadir = args.datadir
    DS = args.DS
    model_files = sorted(glob('tmp/{}.epoch*'.format(DS)))

    graphs = read_graphfile(datadir, DS, max_nodes=None)
    total_len = len(graphs)
    trainer = Trainer(args)
    for path in model_files:
        trainer.model.load_state_dict(torch.load(path))
        trainer.model.eval()
        trainer.model.cuda()
        embeddings = trainer.model.get_embeddings(total_len)
        print(path, evaluate_embedding(datadir, DS, embeddings))
