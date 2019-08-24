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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(epoch, use_unsup_loss):
    global global_step
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    for data, data2 in zip(train_loader, unsup_loader):
        data = data.to(device)
        data2 = data2.to(device)
        optimizer.zero_grad()
        
        pred2 = model(data2)
        teacher_pred2 = teacher_model(data2)

        sup_loss = F.mse_loss(model(data), data.y)
        unsup_loss = F.mse_loss(pred2, teacher_pred2)

        loss = sup_loss + unsup_loss

        loss.backward()
        # print(data)
        # print(loss.item(), data.num_graphs)
        # input()

        sup_loss_all += sup_loss.item() * data.num_graphs
        unsup_loss_all += unsup_loss.item() * data.num_graphs

        loss_all += loss.item() * data.num_graphs
        # tmp.append(loss.item() * data.num_graphs)

        optimizer.step()
        global_step += 1
        update_ema_variables(model, teacher_model, ema_decay, global_step)

        # print(list(model.parameters())[0])
        # input()

    print(sup_loss_all / len(train_loader.dataset), unsup_loss_all / len(train_loader.dataset))
    return loss_all / len(train_loader.dataset)


def test(loader):
    teacher_model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (teacher_model(data) * std - data.y * std).abs().sum().item()  # MAE
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
    ema_decay = .999
    dim = 64
    epochs = 1000
    batch_size = 20
    lamda = args.lamda
    use_unsup_loss = True#args.use_unsup_loss
    separate_encoder = args.separate_encoder

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y[:, target].mean().item()
    std = dataset.data.y[:, target].std().item()
    dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:20000 + args.train_num]
    if use_unsup_loss:
        unsup_train_dataset = dataset[20000:]

    if use_unsup_loss:
        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if use_unsup_loss:
        unsup_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True)
        print('num_features : {}\n'.format(dataset.num_features))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###########################################################
    global_step = 0
    model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    teacher_model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    for param in teacher_model.parameters():
        param.detach_()
    ############################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

    with open('mean-teacher-log', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(target,args.train_num,use_unsup_loss,separate_encoder,args.lamda,args.weight_decay,val_error,test_error))
