import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import KFold
from F_model import *
from tqdm import tqdm
from sklearn import metrics
import warnings


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='promise_nfr.csv', help='The dataset file')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--ckpt', type=str, default='ckpt/', help='The dir that save the model.')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--kf', default=5, type=int, help='Number of kfold')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--id', required=True, help='The id for each training.')
    parser.add_argument('--sen_len', default=18, type=int, help='The length of the input sentence.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--clabel_nb', type=int, default=12, help='quantity of children labels are desired in classification')
    parser.add_argument('--cuda', action='store_true', help='Use GPU to accelerate the training or not.')
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--des_ver', default=1, type=int, help='Use which version of child label description, 1 is the short version, 2 is the long version.')
    opt = parser.parse_args()

    kf = KFold(n_splits=opt.kf)
    dataset = Dataset(opt)
    child_label_des = dataset.getLabelDes()
    sample_len = dataset.__len__()
    if opt.cuda == True:
        child_label_des = child_label_des.cuda()
    # Cross validation
    kf_index = 1
    best_f1 = -1
    for train_index, val_index in kf.split(np.arange(0, sample_len)):

        train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
        val_subset = torch.utils.data.dataset.Subset(dataset, val_index)
        traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)
        testdataloader = torch.utils.data.DataLoader(val_subset, batch_size=1 , shuffle=True, num_workers=opt.workers)
        print("K-fold:{}".format(kf_index))

        kf_index += 1
        model = F_HMN(opt)
        if opt.resume is not None:
            print("\nLoading model from {}...".format(opt.resume))
            model.load_state_dict(torch.load(opt.resume))
        if opt.cuda == True:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.01)
        # Training start
        print('----------------------------Training--------------------------------')
        for epoch in range(1, opt.epoch + 1):
            loss_av = 0.
            sum = 0.
            model.train()
            print(">>epoch:{}".format(epoch))
            for data in tqdm(traindataloader):
                text, parent_label, child_label = data
                if opt.cuda == True:
                    text, parent_label, child_label= text.cuda(), parent_label.cuda(), child_label.cuda()
                optimizer.zero_grad()
                parent_prob, child_prob, b_nf_index, b_f_index = model(text, child_label_des, parent_label)
                loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
                loss_child = 0
                # NF child label loss, child_prob[0] shape: (m, 11)
                if len(child_prob[0]) > 0:

                    loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob[0], child_label[b_nf_index,:opt.clabel_nb - 1])
                # F loss, child_prob[1] shape: (n, 1). m + n = bs
                # if len(child_prob[1]) > 0:
                #     loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob[1], child_label[b_f_index,11:])

                    loss = loss_parent + loss_child
                else:
                    loss = loss_parent
                loss_av += loss.item()
                sum += 1
                loss.backward()
                optimizer.step()
            print('Training Loss:{}'.format(loss_av / sum))
            if epoch % opt.test_freq == 0:
                print('------------------------Evaluation--------------------------------')
                model.eval()
                loss_av = 0.
                parent_prob_sum = []
                child_prob_sum = []
                parent_prob_sum_g = []
                child_prob_sum_g = []
                sum = 0.
                for data in tqdm(testdataloader):
                    text, parent_label, child_label = data
                    if opt.cuda == True:
                        text, parent_label, child_label= text.cuda(), parent_label.cuda(), child_label.cuda()

                    parent_prob, child_prob = model(text, child_label_des, parent_label, mode = 'eval')

                    parent_prob_sum_g.append(parent_label.cpu().squeeze(0).numpy())
                    # if parent_label.cpu().squeeze(0).numpy()[0] == 1:
                    child_prob_sum_g.append(child_label.cpu().squeeze(0).numpy())

                    loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
                    p_p = [0.] * 2
                    p_p[parent_prob.cpu().squeeze(0).argmax(0)] = 1
                    parent_prob_sum.append(p_p)
                    loss_child = 0.

                    # NFR
                    if parent_prob.squeeze()[0] > parent_prob.squeeze()[1]:
                        c_p = [0.] * opt.clabel_nb
                        loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob, child_label.squeeze()[:opt.clabel_nb - 1].unsqueeze(0))
                        c_p[child_prob.cpu().squeeze(0).argmax(0)] = 1
                    #else:
                    # F
                    #     loss_child = 0
                    #     c_p[11] = 1
                        child_prob_sum.append(c_p)
                    else:
                        c_p = [0.] * opt.clabel_nb
                        c_p[-1] = 1
                        child_prob_sum.append(c_p)
                    loss = loss_parent + loss_child
                    loss_av += loss.item()
                    sum += 1
                loss_av /= sum
                parent_prob_sum = np.array(parent_prob_sum)
                child_prob_sum = np.array(child_prob_sum)
                parent_prob_sum_g = np.array(parent_prob_sum_g)
                child_prob_sum_g = np.array(child_prob_sum_g)

                (c_p, c_r, c_f1), c_acc = cal_metric(child_prob_sum_g, child_prob_sum)
                (p_p, p_r, p_f1), p_acc = cal_metric(parent_prob_sum_g, parent_prob_sum)
                (sma_p, sma_r, sma_f1), sacc = cal_metric(np.concatenate((child_prob_sum_g, parent_prob_sum_g), 1), np.concatenate((child_prob_sum, parent_prob_sum), 1))
                print("Parent label : Precision: {} | Recall: {} | F1: {} | Acc : {}".format(p_p, p_r, p_f1, p_acc))
                print("Child label : Precision: {} | Recall: {} | F1: {} | Acc : {}".format(c_p, c_r, c_f1, c_acc))
                print("Summary : Precision: {} | Recall: {} | F1: {} | Acc : {}".format(sma_p, sma_r, sma_f1, sacc))
                print("Evaluation Loss:{}".format(loss_av))
                if sma_f1 > best_f1:
                    best_f1 = sma_f1
                    torch.save(model.state_dict(), os.path.join(opt.ckpt, ''.join([opt.id, '_', str(epoch), '_', str(sma_f1), '.pt'])))
            if (epoch) % 10 == 0:
                adjust_learning_rate(optimizer)

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        a = param_group['lr']
    print('lr:', a)

def cal_metric(y_true, y_pred):
    ma_p, ma_r, ma_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = metrics.accuracy_score(y_true,y_pred)
    return [(ma_p, ma_r, ma_f1), acc]

if __name__ == "__main__":
    main()