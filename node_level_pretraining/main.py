import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import GNNodeEmbedding

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--data_options', default='./data/sample_data')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='classification', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[800], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=32, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()



best_cross_error = 0.
criterion = nn.CrossEntropyLoss()

def main():
    global args, best_cross_error

    # load data
    dataset = CIFData(args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer


    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                      'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in range(len(dataset))]
        '''sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]'''
        _, sample_target, _, _ = collate_pool(sample_data_list)


    # build model
    structures, _, _,_= dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = GNNodeEmbedding(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                )
    linear_pred_atoms = torch.nn.Linear(args.atom_fea_len, 119)
    if args.cuda:
        model.cuda()
        linear_pred_atoms.cuda()
    model_list = [model, linear_pred_atoms]
    # define loss func and optimizer

    if args.optim == 'SGD':
        optimizer_model = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        optimizer_linear_pred_atoms = optim.SGD(linear_pred_atoms.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer_model = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
        optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms]

    scheduler_model = MultiStepLR(optimizer_model, milestones=args.lr_milestones,
                            gamma=0.1)
    scheduler_linear_pred_atoms = MultiStepLR(optimizer_linear_pred_atoms , milestones=args.lr_milestones,
                                  gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epo
        print(epoch)
        train_loss, train_acc_atom =train(train_loader, model_list, optimizer_list)
        print(train_loss, train_acc_atom)

        # evaluate on validation set
        val_loss, val_acc_atom  = validate(val_loader, model_list)
        print(val_loss, val_acc_atom)

        scheduler_model.step()
        scheduler_linear_pred_atoms.step()

        # remember the best mae_eror and save checkpoint
        is_best = val_loss > best_cross_error
        best_cross_error = max(val_loss, best_cross_error)
        save_checkpoint_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            }, is_best)
        save_checkpoint_linear_pred_atoms({
            'epoch': epoch + 1,
            'state_dict': linear_pred_atoms.state_dict()
            }, is_best)


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(train_loader, model_list, optimizer_list):

    # switch to train mode
    model, linear_pred_atoms = model_list
    optimizer_model, optimizer_linear_pred_atoms = optimizer_list
    model.train()
    linear_pred_atoms.train()
    loss_accum = 0
    acc_node_accum = 0
    for step, (input, target, mask_idx,_) in enumerate(train_loader):
        # measure data loading time
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]]
                         )
            mask_idx = mask_idx.cuda(non_blocking=True)

        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4])
        # normalize target

        if args.cuda:
            target = target.cuda(non_blocking=True)
        else:
            target_var = target

        # compute output
        node_rep = model(*input_var)
        mask_idx = mask_idx.squeeze()
        #print('mask',mask_idx)
        node_rep_mask = torch.index_select(node_rep, 0, mask_idx)
        pred_node =linear_pred_atoms(node_rep_mask)
        #print(i)
        #print('output',torch.argmax(pred_node,dim=-1))
        #print('target',target)
        #print(target.size())
        loss = criterion(pred_node.double(), target)
        loss_accum += float(loss.cpu().item())
        #print(pred_node.size())
        acc_node = compute_accuracy(pred_node.detach().cpu(), target.detach().cpu())
        acc_node_accum += acc_node
        # compute gradient and do SGD step
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()


    return loss_accum/len(train_loader) , acc_node_accum/len(train_loader)

def validate(val_loader, model_list, test=False):
    model, linear_pred_atoms = model_list
    # switch to evaluate mode
    model.eval()
    linear_pred_atoms.eval()
    loss_accum = 0
    acc_node_accum = 0
    for step, (input, target, mask_idx,_) in enumerate(val_loader):
        # measure data loading time
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]]
                         )
            mask_ids = mask_idx.cuda(non_blocking=True)

        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4])
        # normalize target

        if args.cuda:
            with torch.no_grad():
                target = target.cuda(non_blocking=True)
        else:
            with torch.no_grad():
                target= Variable(target)

        # compute output
        node_rep = model(*input_var)
        mask_idx = mask_ids.squeeze()
        node_rep_mask = torch.index_select(node_rep, 0, mask_idx)
        pred_node = linear_pred_atoms(node_rep_mask)
        loss = criterion(pred_node.double(), target)
        acc_node = compute_accuracy(pred_node.detach().cpu(), target.detach().cpu())
        acc_node_accum += acc_node
        loss_accum += float(loss.cpu().item())
    return loss_accum/len(val_loader) , acc_node_accum/len(val_loader)





class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint_model(state, is_best, filename='checkpoint_model.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint_linear_pred_atoms(state, is_best, filename='checkpoint_linear_pred_atoms.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'linear_pred_atoms_best.pth.tar')
def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
