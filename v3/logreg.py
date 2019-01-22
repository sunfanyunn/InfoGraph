import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def classify(x, y, validation=False):
    if validation:
        nb_classes = np.unique(y).shape[0]
        # x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        xent = nn.CrossEntropyLoss()


        hid_units = x.shape[1]
        accs = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        for train_index, test_index in kf.split(x, y):
            train_embs, test_embs = x[train_index], x[test_index]
            train_lbls, test_lbls= y[train_index], y[test_index]

            from sklearn.model_selection import train_test_split
            print(train_embs.shape)
            train_embs, valid_embs, train_lbls, valid_lbls = train_test_split(train_embs, train_lbls, test_size=.1, stratify=train_lbls)
            train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
            valid_embs, valid_lbls= torch.from_numpy(valid_embs).cuda(), torch.from_numpy(valid_lbls).cuda()
            test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

            log = LogReg(hid_units, nb_classes)
            log.cuda()
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

            best_val = 0
            test_acc = None
            for it in range(100):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

                logits = log(valid_embs)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == valid_lbls).float() / valid_lbls.shape[0]
                if  acc > best_val:
                    best_val = acc
                # print(acc)

                    logits = log(test_embs)
                    preds = torch.argmax(logits, dim=1)
                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    test_acc = acc
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            print('=================')
            print(test_acc, acc.item())
            print('=================')
            input()
            accs.append(acc.item())

        print('Logreg', np.mean(accs))

        return np.mean(accs)
    else:
        nb_classes = np.unique(y).shape[0]
        x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        xent = nn.CrossEntropyLoss()


        hid_units = x.shape[1]
        accs = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        for train_index, test_index in kf.split(x, y):
            train_embs, test_embs = x[train_index], x[test_index]
            train_lbls, test_lbls= y[train_index], y[test_index]

            log = LogReg(hid_units, nb_classes)
            log.cuda()
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

            best_val = 0
            test_acc = None
            for it in range(100):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc.item())

        # print('Logreg', np.mean(accs))

        return np.mean(accs)
