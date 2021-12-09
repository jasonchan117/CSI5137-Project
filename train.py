import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import KFold
from model import *
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
parser.add_argument('--clabel_nb', type=int, default = 12, help='quantity of children labels are desired in classification')
opt = parser.parse_args()

'''
kf = KFold(n_splits=opt.kf)
dataset = Dataset(opt)
child_label_des = dataset.getLabelDes()
for train_index, val_index in kf.split(np.arange(0, opt.datalen)):

    train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
    val_subset = torch.utils.data.dataset.Subset(dataset, val_index)
    traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(testdataloader, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)

    model = F_HMN(opt)
    model.train()

    for epoch in range(0, opt.epoch):
        for i, data in enumerate(traindataloader, 0):
            text, parent_label, child_label, token_type_ids = data
            y_pred = model(text, token_type_ids)
            loss =
            loss.backward()
'''