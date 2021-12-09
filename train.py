import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import KFold
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'promise_nfr', help='The dataset file')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--ckpt', type=str, default = 'ckpt/', help='The dir save that save the model.')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--kf', default=5, type = int, help = 'Number of kfold')
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--batchsize', default=100, type = int)
parser.add_argument('--datalen', type=int)
parser.add_argument('--workers', type=int, default = 5, help='number of data loading workers')
opt = parser.parse_args()

'''
kf = KFold(n_splits=opt.kf)
for train_index, val_index in kf.split(np.arange(0, opt.datalen)):
    train_subset = torch.utils.data.dataset.Subset(Dataset(opt), train_index)
    val_subset = torch.utils.data.dataset.Subset(Dataset(opt), val_index)
    dataloader = torch.utils.data.DataLoader(train_subset, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(testdataloader, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)

    model = 
    model.train()

    for epoch in range(0, opt.epoch):
        for i, data in enumerate(dataloader, 0):
            description, inputs = data
            y_pred = model(description, inputs)
            loss =
            loss.backward()
'''