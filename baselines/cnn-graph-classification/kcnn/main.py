# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import shutil
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

# Our Modules
from utils import compute_nystrom, create_train_val_test_loaders, save_checkpoint, AverageMeter
from model import CNN

# Argument parser
parser = argparse.ArgumentParser(description='Kernel Graph CNN')

parser.add_argument('--dataset', default='IMDB-BINARY', help='dataset name')
parser.add_argument('--community-detection', default='louvain', help='community detection method')
parser.add_argument('--checkpoint-dir', default='./checkpoint/kcnn/', help='path to latest checkpoint')

# Optimization Options
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='Input batch size for training (default: 20)')
parser.add_argument('--n-filters', type=int, default=128, metavar='N', help='Number of filters (default: 64)')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='Size of hidden layer (default: 128)')
parser.add_argument('--d', type=int, default=200, metavar='N', help='Dimensionality of graph embeddings (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate (default: 1e-4)')
parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR-DECAY', help='Learning rate decay factor (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S', help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')

# i/o
parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='How many batches to wait before logging training status')



def main():
    global args
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    unlabeled_datasets = ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "SYNTHETIC", "raw-gitgraph"]
    if args.dataset in unlabeled_datasets:
        use_node_labels = False
        from graph_kernels import sp_kernel, wl_kernel
    else:
        use_node_labels = True
        from graph_kernels_labeled import sp_kernel, wl_kernel


    kernels = [wl_kernel]
    n_kernels = len(kernels)

    print('Computing graph maps')
    Q, subgraphs, labels, shapes = compute_nystrom(args.dataset, use_node_labels, args.d, args.community_detection, kernels)

    M=np.zeros((shapes[0],shapes[1],n_kernels))
    for idx,k in enumerate(kernels):
        M[:,:,idx]=Q[idx]

    Q=M

    # Binarize labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Build vocabulary
    max_n_communities = max([len(x.split(" ")) for x in subgraphs])
    x = np.zeros((len(subgraphs), max_n_communities), dtype=np.int32)
    for i in range(len(subgraphs)):
        communities = subgraphs[i].split()
        for j in range(len(communities)):
            x[i,j] = int(communities[j])

    print(x[0,:])

    kf = StratifiedKFold(n_splits=10, random_state=None)
    kf.shuffle=True
    accs=[];
    it = 0

    print('Starting cross-validation')

    for train_index, test_index in kf.split(x, y):
        it += 1
        best_acc1 = 0

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

        train_loader, val_loader, test_loader = create_train_val_test_loaders(Q, x_train, x_val, x_test, y_train, y_val, y_test, args.batch_size)

        print('\tCreate model')
        model = CNN(input_size=args.n_filters, hidden_size=args.hidden_size, n_classes=np.unique(y).size, d=args.d, n_kernels=n_kernels, max_n_communities=max_n_communities)

        print('Optimizer')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        criterion = nn.CrossEntropyLoss()

        evaluation = lambda output, target: torch.sum(output.eq(target)) / target.size()[0]

        lr = args.lr
        lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])
		  
        if os.path.isdir(args.checkpoint_dir):
        	shutil.rmtree(args.checkpoint_dir)
        
        os.makedirs(args.checkpoint_dir)

        print('Check cuda')
        if args.cuda:
            print('\t* Cuda')
            model = model.cuda()
            criterion = criterion.cuda()

        # Epoch for loop
        for epoch in range(0, args.epochs):

            if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
                lr -= lr_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, evaluation)

            # evaluate on test set
            acc1 = validate(val_loader, model, criterion, evaluation)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.checkpoint_dir)

        # get the best checkpoint and test it with test set
        best_model_file = os.path.join(args.checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

        # For testing
        acc = validate(test_loader, model, criterion, evaluation)
        print("Accuracy at iteration " + str(it) + ": " + str(acc))
        accs.append(acc)
    print("Average accuracy: ", np.mean(accs))
    print("std: ", np.std(accs))


def train(train_loader, model, criterion, optimizer, epoch, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_accuracy = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (graph, target) in enumerate(train_loader):

        # Prepare input data
        if args.cuda:
            graph, target = graph.cuda(), target.cuda()
        graph, target = Variable(graph), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute output
        output = model(graph)
        
        train_loss = criterion(output, target)

        # Logs
        losses.update(train_loss.data[0], graph.size(0))
        
        _, predicted = torch.max(output, 1)
        avg_accuracy.update(evaluation(predicted.data, target.data), graph.size(0))

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Avg accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, acc=avg_accuracy))

    print('Epoch: [{0}] Avg accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, acc=avg_accuracy, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (graph, target) in enumerate(val_loader):

        # Prepare input data
        if args.cuda:
            graph, target = graph.cuda(), target.cuda()
        graph, target = Variable(graph), Variable(target)

        # Compute output
        output = model(graph)
        
        # Logs
        losses.update(criterion(output, target).data[0], graph.size(0))
        _, predicted = torch.max(output, 1)
        avg_accuracy.update(evaluation(predicted.data, target.data), graph.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Average accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=avg_accuracy))

    print(' * Average accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(acc=avg_accuracy, loss=losses))

    return avg_accuracy.avg

    
if __name__ == '__main__':
    main()
