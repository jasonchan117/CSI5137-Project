import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import KFold
from model import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'promise_nfr.csv', help='The dataset file')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--ckpt', type=str, default = 'ckpt/', help='The dir that save the model.')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--kf', default=5, type = int, help = 'Number of kfold')
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--batchsize', default=8, type = int)
parser.add_argument('--sen_len', default=18, type=int, help = 'The length of the input sentence.')
parser.add_argument('--workers', type=int, default = 5, help='number of data loading workers')
parser.add_argument('--clabel_nb', type=int, default = 12, help='quantity of children labels are desired in classification')
parser.add_argument('--cuda', action= 'store_true', help = 'Use GPU to accelerate the training or not.')
opt = parser.parse_args()


kf = KFold(n_splits=opt.kf)
dataset = Dataset(opt)
child_label_des, child_label_len = dataset.getLabelDes()
# Cross validation
for train_index, val_index in kf.split(np.arange(0, opt.datalen)):

    train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
    val_subset = torch.utils.data.dataset.Subset(dataset, val_index)
    traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(testdataloader, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)

    model = F_HMN(opt)
    if opt.cuda == True:
        model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # Training start
    for epoch in range(0, opt.epoch):
        for i, data in enumerate(traindataloader, 0):
            text, parent_label, child_label, token_type_ids = data
            if opt.cuda == True:
                text, parent_label, child_label, token_type_ids = text.cuda(), parent_label.cuda(), child_label.cuda(), token_type_ids.cuda()
            optimizer.zero_grad()
            parent_prob, child_prob, b_nf_index, b_f_index = model(text, token_type_ids, child_label_des, child_label_len, parent_label)
            loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
            loss_child = 0
            # NF child label loss, child_prob[0] shape: (m, 11)
            if len(child_prob[0]) > 0:
                loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob[0], child_label[b_nf_index])
            # F loss, child_prob[1] shape: (n, 1). m + n = bs
            if len(child_prob[1]) > 0:
                loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob[1], child_label[b_f_index])

            loss = loss_parent + loss_child
            loss.backward()
            optimizer.step()