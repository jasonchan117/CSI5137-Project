import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from sklearn.model_selection import StratifiedKFold as KFold
from F_model import *
from tqdm import tqdm
from sklearn import metrics
import warnings
import sys

class Evaluation:

    def __init__(self, opt, label_names):
        self.opt = opt
        self.best_f1 = [-1.] * opt.kf
        self.best_parent_prob_sum = [[]] * opt.kf
        self.best_child_prob_sum = [[]] * opt.kf
        self.best_parent_prob_sum_g = [[]] * opt.kf
        self.best_child_prob_sum_g = [[]] * opt.kf
        self.label_names = label_names

    def record(self,parent_prob_sum, child_prob_sum, parent_prob_sum_g, child_prob_sum_g, eval_log, fold_index):
        if self.opt.model_type != 'Bert_c':
            (p_p, p_r, p_f1), p_acc = self.cal_metric(parent_prob_sum_g, parent_prob_sum, 'macro')
            p_flag = p_f1
            # Print and record the parent label overall metrics
            print("Parent label Summary : Precision: {} | Recall: {} | F1: {} | Acc : {}".format(p_p, p_r, p_f1, p_acc))
            eval_log.write(
                "Parent label Summary: Precision: {} | Recall: {} | F1: {} | Acc : {}\n".format(p_p, p_r, p_f1, p_acc))
            print("Parent label per class :")
            (p_p, p_r, p_f1), p_acc = self.cal_metric(parent_prob_sum_g, parent_prob_sum, None)
            # Print and record the per parent label metrics
            eval_log.write("NFR: Precision: {} | Recall: {} | F1: {}\n".format(p_p[0], p_r[0], p_f1[0]))
            eval_log.write("F: Precision: {} | Recall: {} | F1: {}\n".format(p_p[1], p_r[1], p_f1[1]))
            eval_log.write("Parent Label Accuracy:{}\n".format(p_acc))
            print("NFR: Precision: {} | Recall: {} | F1: {}".format(p_p[0], p_r[0], p_f1[0]))
            print("F: Precision: {} | Recall: {} | F1: {}".format(p_p[1], p_r[1], p_f1[1]))
            print("Parent label Accuracy:{}".format(p_acc))
        print()
        if self.opt.model_type != 'Bert_p' :
            if self.opt.model_type == 'HMN':
                start_num = 1
                temp = child_prob_sum_g
            else:
                start_num = 2
                temp = child_prob_sum_g[:,1:]
            (c_p, c_r, c_f1), c_acc = self.cal_metric(temp, child_prob_sum, 'macro')
            c_flag = c_f1
            # Print and record the child label overall metrics
            eval_log.write(
                "Child label Summary: Precision: {} | Recall: {} | F1: {} | Acc : {}\n".format(c_p, c_r, c_f1, c_acc))
            print("Child label Summary: Precision: {} | Recall: {} | F1: {} | Acc : {}".format(c_p, c_r, c_f1, c_acc))

            print("Child label per class :")
            (c_p, c_r, c_f1), c_acc = self.cal_metric(temp, child_prob_sum, None)
            # Print and record the per child label metircs.
            for ind, name in enumerate(self.label_names[start_num:self.opt.clabel_nb + 1]):
                eval_log.write(
                    '{}: Precision: {} | Recall: {} | F1: {}\n'.format(name, c_p[ind], c_r[ind], c_f1[ind]))
                print('{}: Precision: {} | Recall: {} | F1: {}'.format(name, c_p[ind], c_r[ind], c_f1[ind]))
            if self.opt.clabel_nb != 12:
                eval_log.write(
                    '{}: Precision: {} | Recall: {} | F1: {}\n'.format('OTH', c_p[-1], c_r[-1], c_f1[-1]))
                print('{}: Precision: {} | Recall: {} | F1: {}'.format('OTH', c_p[-1], c_r[-1], c_f1[-1]))
            eval_log.write("Child label Accuracy:{}\n".format(c_acc))
            print("Child label Accuracy:{}".format(c_acc))

        if self.opt.model_type == 'Bert_c' and c_flag > self.best_f1[fold_index]:
            self.best_f1[fold_index] = c_flag
            self.best_child_prob_sum[fold_index] = child_prob_sum
            self.best_child_prob_sum_g[fold_index] = child_prob_sum_g[:,1:]
            return c_flag
        if self.opt.model_type == 'Bert_p' and p_flag > self.best_f1[fold_index]:
            self.best_f1[fold_index] = p_flag
            self.best_parent_prob_sum[fold_index] = parent_prob_sum
            self.best_parent_prob_sum_g[fold_index] = parent_prob_sum_g
            return p_flag
        if self.opt.model_type == 'HMN' and c_flag > self.best_f1[fold_index] :
            self.best_f1[fold_index] = c_flag
            self.best_child_prob_sum[fold_index] = child_prob_sum
            self.best_child_prob_sum_g[fold_index] = child_prob_sum_g[:,1:]
            self.best_parent_prob_sum[fold_index] = parent_prob_sum
            self.best_parent_prob_sum_g[fold_index] = parent_prob_sum_g
            return c_flag
        return False
    def cal_overall_metric(self, eval_log):

        for ind, i in enumerate(self.best_parent_prob_sum):
            if ind == 0:
                if self.opt.model_type != 'Bert_c':
                    parent_all = self.best_parent_prob_sum[ind]
                    parent_all_g = self.best_parent_prob_sum_g[ind]
                if self.opt.model_type != 'Bert_p':
                    child_all = self.best_child_prob_sum[ind]
                    child_all_g = self.best_child_prob_sum_g[ind]
            else:
                if self.opt.model_type != 'Bert_c':
                    parent_all = np.concatenate((parent_all, self.best_parent_prob_sum[ind]), 0)
                    parent_all_g = np.concatenate((parent_all_g, self.best_parent_prob_sum_g[ind]), 0)
                if self.opt.model_type != 'Bert_p':
                    child_all = np.concatenate((child_all, self.best_child_prob_sum[ind]), 0)
                    child_all_g = np.concatenate((child_all_g, self.best_child_prob_sum_g[ind]), 0)
        print("********************************************************************")
        eval_log.write("********************************************************************\n")
        print('Overall Performance among all folds:')
        eval_log.write('Overall Performance among all folds:')
        if self.opt.model_type != 'Bert_c':
            (p_p, p_r, p_f1), p_acc = self.cal_metric(parent_all_g, parent_all, None)
            eval_log.write("NFR: Precision: {} | Recall: {} | F1: {}\n".format(p_p[0], p_r[0], p_f1[0]))
            eval_log.write("F: Precision: {} | Recall: {} | F1: {}\n".format(p_p[1], p_r[1], p_f1[1]))
            eval_log.write("Parent Label Accuracy:{}\n".format(p_acc))
            print("NFR: Precision: {} | Recall: {} | F1: {}".format(p_p[0], p_r[0], p_f1[0]))
            print("F: Precision: {} | Recall: {} | F1: {}".format(p_p[1], p_r[1], p_f1[1]))
            print("Parent label Accuracy:{}".format(p_acc))
        if self.opt.model_type != 'Bert_p':
            (c_p, c_r, c_f1), c_acc = self.cal_metric(child_all_g, child_all, None)
            for ind, name in enumerate(self.label_names[2:self.opt.clabel_nb + 1]):
                eval_log.write(
                    '{}: Precision: {} | Recall: {} | F1: {}\n'.format(name, c_p[ind], c_r[ind], c_f1[ind]))
                print('{}: Precision: {} | Recall: {} | F1: {}'.format(name, c_p[ind], c_r[ind], c_f1[ind]))
            if self.opt.clabel_nb != 12:
                eval_log.write(
                    '{}: Precision: {} | Recall: {} | F1: {}\n'.format('OTH', c_p[-1], c_r[-1], c_f1[-1]))
                print('{}: Precision: {} | Recall: {} | F1: {}'.format('OTH', c_p[-1], c_r[-1], c_f1[-1]))
            eval_log.write("Child label Accuracy:{}\n".format(c_acc))
            print("Child label Accuracy:{}".format(c_acc))


    def cal_metric(self, y_true, y_pred, average='macro'):
        ma_p, ma_r, ma_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)
        acc = metrics.accuracy_score(y_true, y_pred)
        return [(ma_p, ma_r, ma_f1), acc]