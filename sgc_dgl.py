"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SGConv
import random


def set_random_seed(seed=1024):
    print('>> set random seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def plot_curve(train_acc_npy, test_acc_npy, dataset):
    import matplotlib.pyplot as plt
    plt.plot(train_acc_npy, label="Train acc")
    plt.plot(test_acc_npy, label="Test acc")
    plt.title('{}'.format(dataset))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.axis([-3, 303, 0.5, 1])
    plt.legend(loc="upper left")
    plt.savefig('./loss_curve/{}_loss.eps'.format(dataset))
    plt.show()


def main(args):
    # load and preprocess dataset
    train_acc_list = []
    test_acc_list = []
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
        print('>> no use GPU')
    else:
        cuda = True
        print('>> using GPU ...')
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())

    # create SGC model
    model = SGConv(in_feats,
                   n_classes,
                   k=2,
                   cached=True,
                   bias=args.bias)

    if cuda: model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()
    # loss_fcn = FocalLoss(gamma=0)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)  # only compute the train set
        print(torch.nonzero(train_mask))
        print(logits[train_mask].size())
        exit()
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_mask)
        test_acc = evaluate(model, g, features, labels, test_mask)
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.6f} | Val accuracy {:.6f} | Test accuracy {:.6f} | ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(),
                acc, test_acc, n_edges / np.mean(dur) / 1000))
        train_acc_list.append(acc)
        test_acc_list.append(test_acc)
    plot_curve(train_acc_list, test_acc_list, args.dataset)
    # print()
    # acc = evaluate(model, g, features, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pubmed')
    # register_data_args(parser
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.2,
                        help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
                        help="flag to use bias")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-5,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    args.dataset = 'pubmed'
    print(args)
    set_random_seed(seed=128)
    main(args)