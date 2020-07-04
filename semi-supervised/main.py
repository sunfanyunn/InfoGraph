import os
import sys
import os.path as osp
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    if use_unsup_loss:
        for data, data2 in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()

            sup_loss = F.mse_loss(model(data), data.y)
            unsup_loss = model.unsup_loss(data2)
            if separate_encoder:
                unsup_sup_loss = model.unsup_sup_loss(data2)
                loss = sup_loss + unsup_loss + unsup_sup_loss * lamda
            else:
                loss = sup_loss + unsup_loss * lamda

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            if separate_encoder:
                unsup_sup_loss_all += unsup_sup_loss.item()
            loss_all += loss.item() * data.num_graphs

            optimizer.step()

        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
        else:
            print(sup_loss_all, unsup_loss_all)
        return loss_all / len(train_loader.dataset)
    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            sup_loss = F.mse_loss(model(data), data.y)
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from model import Net
    from arguments import arg_parse 
    args = arg_parse()

    target = args.target
    dim = 64
    epochs = 500
    batch_size = 20
    lamda = args.lamda
    use_unsup_loss = args.use_unsup_loss
    separate_encoder = args.separate_encoder

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()
    print('num_features : {}\n'.format(dataset.num_features))

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y[:, target].mean().item()
    std = dataset.data.y[:, target].std().item()
    dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    # print(type(dataset[0]))
    # print(type(dataset.data.x)) #tensor
    # print(type(dataset.data.y)) #tensor
    # input()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:20000+args.train_num]

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if use_unsup_loss:
        unsup_train_dataset = dataset[20000:]
        unsup_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True)

        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

    val_error = test(val_loader)
    test_error = test(test_loader)
    print('Epoch: {:03d}, Validation MAE: {:.7f}, Test MAE: {:.7f},'.format(0, val_error, test_error))

    best_val_error = None
    for epoch in range(1, epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch, use_unsup_loss)
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            print('Update')
            test_error = test(test_loader)
            best_val_error = val_error


        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},'.format(epoch, lr, loss, val_error, test_error))

    with open('supervised.log', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(target,args.train_num,use_unsup_loss,separate_encoder,args.lamda,args.weight_decay,val_error,test_error))

    try:
        torch.save(model, 'saved_models/{}.model'.format(target))
    except Exception as e:
        print(e)
