import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(fn),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    err, loss_ls = 0.0, []
    if opt is not None:
        model.train()
        for data, target in dataloader:
            opt.reset_grad()
            logits = model(data)
            err += np.sum(logits.numpy().argmax(1) != target.numpy())
            loss = loss_func(logits, target)
            loss.backward()
            loss_ls.append(loss.numpy())
            opt.step()
    else:
        model.eval()
        for data, target in dataloader:
            logits = model(data)
            err += np.sum(logits.numpy().argmax(1) != target.numpy())
            loss = loss_func(logits, target)
            loss_ls.append(loss.numpy())
    return err / len(dataloader.dataset), np.mean(loss_ls)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # model & optimizer
    mlp_res_net = MLPResNet(28 * 28, hidden_dim, num_classes=10)
    opt = optimizer(mlp_res_net.parameters(), lr=lr, weight_decay=weight_decay)
    # data
    train_dataset = ndl.data.MNISTDataset(f'{data_dir}/train-images-idx3-ubyte.gz',
                                          f'{data_dir}/train-labels-idx1-ubyte.gz')
    test_dataset = ndl.data.MNISTDataset(f'{data_dir}/t10k-images-idx3-ubyte.gz',
                                         f'{data_dir}/t10k-labels-idx1-ubyte.gz')
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size)
    # train
    train_err, train_loss = 0, 0
    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, mlp_res_net, opt)
    # test
    test_err, test_loss = epoch(test_dataloader, mlp_res_net)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
