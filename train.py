import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import StratifiedKFold as KFold
from F_model import *
from tqdm import tqdm
from pytorch_transformers import AdamW
from sklearn import metrics
import warnings
import sys
from eval import *
def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='promise_nfr.csv', help='The dataset file')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='The dir that save the model.')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--kf', default=5, type=int, help='Number of kfold')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--id', required=True, help='The id for each training.', type=str)
    parser.add_argument('--sen_len', default=50, type=int, help='The length of the input sentence.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--clabel_nb', type=int, default=12, help='quantity of children labels are desired in classification, the F is included in this valuable.')
    parser.add_argument('--cuda', action='store_true', help='Use GPU to accelerate the training or not.')
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--des_ver', default=2, type=int, help='Use which version of child label description, 1 is the short version, 2 is the long version.')
    parser.add_argument('--wd', default=0.01, type=float, help='Weight decay.')
    parser.add_argument('--pretrain', default = 'large', help = 'Pretrain bert version: large, base')
    parser.add_argument('--model_type', default = 'HMN', help = 'Use which model to do the task.(HMN, Bert_p: Parent label classifier, Bert_c: Child label classifier)')
    opt = parser.parse_args()
    if not os.path.exists(opt.ckpt):
        os.makedirs(opt.ckpt)
    if not os.path.exists(os.path.join(opt.ckpt, opt.id)):
        os.makedirs(os.path.join(opt.ckpt, opt.id))

    kf = KFold(n_splits=opt.kf)
    dataset = Dataset(opt)
    child_label_des = dataset.getLabelDes()
    sample_len = dataset.__len__()
    if opt.cuda == True:
        child_label_des = child_label_des.cuda()
    # Cross validation
    kf_index = 1
    Y = []
    for i in dataset.c_labels:
        cl = opt.clabel_nb if opt.clabel_nb == 12 else opt.clabel_nb + 1
        for ind, j in enumerate(i[:cl]):
            if j == 1:
                Y.append(ind)
                break
    Y = np.array(Y)
    eval_entity = Evaluation(opt, dataset.label_names)
    # K split
    for train_index, val_index in kf.split(np.arange(0, sample_len),Y):

        train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
        val_subset = torch.utils.data.dataset.Subset(dataset, val_index)
        traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=opt.batchsize , shuffle=True, num_workers=opt.workers)
        testdataloader = torch.utils.data.DataLoader(val_subset, batch_size=1 , shuffle=True, num_workers=opt.workers)
        print("K-fold:{}".format(kf_index))
        kf_index += 1
        if opt.model_type == 'HMN':
            model = F_HMN(opt)
        else:
            model = bertModel(opt)
        # Loading models
        if opt.resume is not None:
            print("\nLoading model from {}...".format(opt.resume))
            model.load_state_dict(torch.load(opt.resume))
        if opt.cuda == True:
            model = model.cuda()

        optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        # Training start
        print('----------------------------Training--------------------------------')
        for epoch in range(1, opt.epoch + 1):
            loss_av = 0.
            e_sum = 0.
            model.train()
            print(">>epoch:{}".format(epoch))
            for data in tqdm(traindataloader):
                text, parent_label, child_label = data
                if opt.cuda == True:
                    text, parent_label, child_label= text.cuda(), parent_label.cuda(), child_label.cuda()
                optimizer.zero_grad()

                if opt.model_type =='HMN':
                    parent_prob, child_prob, b_nf_index, b_f_index = model(text, child_label_des, parent_label)
                    loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
                    loss_child = 0
                    # NF child label loss, child_prob[0] shape: (m, 11)
                    if len(child_prob[0]) > 0:

                        loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob[0], child_label[b_nf_index,1:])
                        loss = loss_parent + loss_child
                    else:
                        loss = loss_parent
                elif opt.model_type =='Bert_c':
                    child_prob = model(text)
                    # Only on the child label. If clabel_nb is not equal to 12, then it will include a other label.
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(child_prob, child_label[:, 1:])
                else:
                    # Model type == Bert_p
                    parent_prob = model(text)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
                loss_av += loss.item()
                e_sum += 1
                loss.backward()
                optimizer.step()
            print('Training Loss:{}'.format(loss_av / e_sum))


            if epoch % opt.test_freq == 0:
                print('------------------------Evaluation--------------------------------')
                model.eval()
                # The arrays that store the ground truth and predicted results.
                loss_av = 0.
                parent_prob_sum = []
                child_prob_sum = []
                parent_prob_sum_g = []
                child_prob_sum_g = []
                e_sum = 0.
                for data in tqdm(testdataloader):
                    text, parent_label, child_label = data
                    parent_prob_sum_g.append(parent_label.cpu().squeeze(0).numpy())
                    child_prob_sum_g.append(child_label.cpu().squeeze(0).numpy())
                    p_p = [0.] * 2

                    if opt.cuda == True:
                        text, parent_label, child_label= text.cuda(), parent_label.cuda(), child_label.cuda()
                    if opt.model_type == 'HMN':
                        if opt.clabel_nb == 12:
                            c_p = [0.] * opt.clabel_nb
                        else:
                            c_p = [0.] * (opt.clabel_nb + 1)
                        parent_prob, child_prob = model(text, child_label_des, parent_label, mode = 'eval')
                        loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)
                        p_p[parent_prob.cpu().squeeze(0).argmax(0)] = 1
                        parent_prob_sum.append(p_p)
                        loss_child = 0.
                        # NFR
                        if parent_prob.squeeze()[0] > parent_prob.squeeze()[1]:
                            loss_child += torch.nn.functional.binary_cross_entropy_with_logits(child_prob, child_label.squeeze()[1:].unsqueeze(0))
                            c_p[child_prob.cpu().squeeze(0).argmax(0) + 1] = 1
                            child_prob_sum.append(c_p)
                        else:
                            c_p[0] = 1
                            child_prob_sum.append(c_p)
                        loss = loss_parent + loss_child
                    elif opt.model_type == 'Bert_c':
                        if opt.clabel_nb == 12:
                            c_p = [0.] * (opt.clabel_nb - 1)
                        else:
                            c_p = [0.] * opt.clabel_nb
                        child_prob = model(text)
                        c_p[child_prob.cpu().squeeze(0).argmax(0)] = 1
                        child_prob_sum.append(c_p)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(child_prob, child_label[:,1:])
                    else:
                        # Bert_p
                        parent_prob = model(text)
                        p_p[parent_prob.cpu().squeeze(0).argmax(0)] = 1
                        parent_prob_sum.append(p_p)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(parent_prob, parent_label)

                    loss_av += loss.item()
                    e_sum += 1
                loss_av /= e_sum

                parent_prob_sum = np.array(parent_prob_sum)
                child_prob_sum = np.array(child_prob_sum)
                parent_prob_sum_g = np.array(parent_prob_sum_g)
                child_prob_sum_g = np.array(child_prob_sum_g)


                eval_log = open(os.path.join(opt.ckpt, opt.id, 'eval_log.txt'), 'a')
                eval_log.write("-----------------------------")
                eval_log.write("Fold #%d   Epoch #%d\n" % (kf_index-1, epoch))

                store_flag = eval_entity.record(parent_prob_sum, child_prob_sum, parent_prob_sum_g, child_prob_sum_g, eval_log, kf_index - 2)

                if store_flag != False:
                    torch.save(model.state_dict(),os.path.join(opt.ckpt, opt.id, ''.join([ 'Fold','_',str(kf_index-1),'_', 'epoch','_',str(epoch), '_', str(store_flag), '.pt'])))

                print("Evaluation Loss:{}".format(loss_av))
                eval_log.close()

            if (epoch) % 10 == 0:
                adjust_learning_rate(optimizer)

    eval_log = open(os.path.join(opt.ckpt, opt.id, 'eval_log.txt'), 'a')
    eval_entity.cal_overall_metric(eval_log)
    eval_log.close()

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        a = param_group['lr']
    print('lr update:', a)

if __name__ == "__main__":
    main()