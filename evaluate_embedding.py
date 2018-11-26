from data_utils import read_graphfile
import numpy as np
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


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
    print(np.mean(accuracies))
    return np.mean(accuracies)

    # print(labels)
    # np.random.shuffle(labels)
    # print(labels)
    # input()

    # params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
    # params = {'C':[10]}
    # classifier = GridSearchCV(SVC(), params, cv=10, scoring='accuracy', verbose=2)
    # classifier.fit(embeddings, labels)
    # print('best classifier model\'s hyperparamters', classifier.best_params_)

    # clf = GridSearchCV(svc, parameters, cv=5)
    # clf.fit(vecs, labels)
    # classifier = SVC(C=10)
    # scores = cross_val_score(classifier, embeddings, labels, cv=10)
    # classifier.fit(embeddings, labels)
    # print(classifier.predict(embeddings))

    # print(np.mean(scores))
    # print(scores)

if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
