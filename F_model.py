from model.attentions.RSABlock import *
from model.layers.FCLayer import *
from model.model_utils import LayerNorm
from transformers import BertTokenizer, BertForMultipleChoice
from transformers import AdamW
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class F_HMN(nn.Module):
    def __init__(self, opt):
        super(F_HMN, self).__init__()
        # The bert model object
        self.bert_description = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-'+ opt.pretrain +'-cased')
        self.bert_text = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-'+ opt.pretrain +'-cased')
        self.opt = opt
        self.RSANModel = RSANModel(opt)
        if self.opt.pretrain == 'base':
            self.parent_fc = FCLayer(2*768, 2, type="deep")
            self.nf_fc = FCLayer(2 * 768, self.opt.clabel_nb - 1)
        else:
            self.parent_fc = FCLayer(2*1024, 2, type="deep")
            self.nf_fc = FCLayer(2 * 1024, self.opt.clabel_nb - 1)
        self.coatt_nf = RSANModel(opt)
        self.coatt_f = RSANModel(opt)



    def cal(self,input_list, last_hidden):
        """
        use CosineSimilarity to calculate the reputation
        :param input_list
        :param last_hidden
        :return:father
        :return:output_list(one father class's subclasses)
        """
        #B is the num of subclasses
        #[B,H]--[B*B,L,H]
        b = (last_hidden.unsqueeze(1)).repeat(len(last_hidden), input_list.size()[1], 1)
        #[B,L,H]--[B*B,L,H]
        a = [(law).unsqueeze(0).repeat(len(input_list), 1, 1) for law in input_list]
        a = torch.cat(a)

        # #[B*B,L,L]
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        #[B*B,L,1]
        #print(a.size(), b.size())
        cosine = cos(a,b).unsqueeze(2)
        # [B*B,L,H]*[B*B,L,1]
        attention = a*cosine
        # [B, B, L, H]
        if self.opt.pretrain == 'base':
            part_list = attention.view(len(input_list), len(input_list), -1, 768)
        else:
            part_list = attention.view(len(input_list), len(input_list), -1, 1024)
        # [B,B,L,H]----[B,1,L,H]-----[B,L,H]  all subclasses's attention
        avgpool = F.avg_pool3d(part_list, (len(input_list), 1, 1)).squeeze()

        # weight = torch.div(avgpool,input_list)

        output_list = input_list-avgpool
        father = torch.cat([avgpool])
        return father, output_list

    def forward(self, text, child_label_des, p_label, mode = 'train'):
        # child_label_des: (12, max), child_label_len: (12, )
        # parent_label: (bs, 2)

        # print(text.size(), child_label_des.size(),p_label.size())
        des_output = self.bert_description(child_label_des)
        # (12, max, 768)
        des_per_embed = des_output[0]
        # (12, 768)
        des_embed = des_output[1]

        nf_embed = des_embed[1:] #
        f_embed = des_embed[0]

        nf_per_embed = des_per_embed[1:]
        f_per_embed = des_per_embed[0]

        nf_output = self.cal(nf_per_embed, nf_embed)

        f_output = self.cal(f_per_embed.unsqueeze(0), f_embed.unsqueeze(0))
        parent_label = torch.cat([nf_output[0], f_output[0].unsqueeze(0)])
        all_list = [F.max_pool1d(nf_output[1].transpose(1, 2), nf_output[1].transpose(1, 2).size(2)).squeeze(2), F.max_pool1d(f_output[1].transpose(1, 2), f_output[1].transpose(1, 2).size(2)).squeeze(2)]
        # parent_label shape:(12, 768)
        label_des = F.max_pool1d(parent_label.transpose(1, 2), parent_label.transpose(1, 2).size(2)).squeeze(2)
        # (bs, 12, 768)
        label_repeat_out = label_des.repeat((text.size(0), 1, 1))
        # (bs, 18, 768)
        text_embed = self.bert_text(text)[0]
        # output_feature shape (bs, 2 x 768)
        output_feature = self.RSANModel(text_embed, label_repeat_out)
        # shape (bs, 2)
        parent_prob = self.parent_fc(output_feature)

        if mode == 'train':
            # Get the index of F case and NF case in one batch
            b_f_index = []
            b_nf_index = []
            for index, i in enumerate(p_label):
                if i[1].item() == 1.:
                    b_f_index.append(index)
                else:
                    b_nf_index.append(index)
            # NF classification
            child_prob = []
            if len(b_nf_index) != 0:
                NF_child_label_prob = self.coatt_nf(text_embed[b_nf_index], label_des[1:].repeat(len(b_nf_index), 1, 1))
                child_prob.append(self.nf_fc(NF_child_label_prob))
            else:
                child_prob.append([])
            # F classification
            # if len(b_f_index) != 0:
            #     F_child_label_prob = self.coatt_f(text_embed[b_f_index], label_des[11:].repeat(len(b_f_index), 1, 1))
            #     child_prob.append(self.f_fc(F_child_label_prob))
            # else:
            child_prob.append([])

            return parent_prob, child_prob, b_nf_index, b_f_index
        else:
            # NFR
            if F.sigmoid(parent_prob).squeeze(0)[0] > F.sigmoid(parent_prob).squeeze(0)[1]:
                return F.sigmoid(parent_prob), F.sigmoid(self.nf_fc(self.coatt_nf(text_embed, label_des[1:].unsqueeze(0))))
            else:
            # F
                return F.sigmoid(parent_prob), 0





class RSANModel(nn.Module):
    def __init__(self, opt):
        super(RSANModel, self).__init__()
        if opt.pretrain == 'base':
            self.norm = LayerNorm(768)
        else:
            self.norm = LayerNorm(1024)
        self.rsa_blocks = RSABlock(opt)
        self.opt = opt
    def forward(self, inputs, label_inputs):

        """
        :param inputs
        :param inputs_length
        :param label_inputs
        :param label_inputs_length
        :return:output_feature
        """
        if self.opt.cuda == True:
            inputs_length = torch.LongTensor([self.opt.sen_len for i in range(inputs.size(0))]).cuda()
            docs_len = Variable(torch.LongTensor([label_inputs.size(1)] * inputs_length.size(0))).cuda()
        else:
            inputs_length = torch.LongTensor([self.opt.sen_len for i in range(inputs.size(0))])
            docs_len = Variable(torch.LongTensor([label_inputs.size(1)] * inputs_length.size(0)))
        FactAoA = inputs
        LabelAoA = label_inputs

        # (B,L,H), (B,LS,H)
        FactAoA, LabelAoA = self.rsa_blocks(FactAoA, inputs_length, LabelAoA, docs_len)
        FactAoA = self.norm(FactAoA)
        LabelAoA = self.norm(LabelAoA)

        # simple version
        # (B, L, H) -> (B, H)
        FactAoA_output = torch.mean(FactAoA,dim=1)
        # (B, LS, H) -> (B, H)
        LabelAoA_output = torch.mean(LabelAoA,dim=1)
        # (B, H) + (B, H) -> (B, 2H)
        output_feature = torch.cat((FactAoA_output, LabelAoA_output), dim=-1)

        return output_feature


class bertModel(nn.Module):
    def __init__(self, opt):
        super(bertModel, self).__init__()
        self.opt = opt
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-' + opt.pretrain + '-cased')
        if opt.pretrain == 'base':
            hidden = 768
        else:
            hidden = 1024
        self.dropout = nn.Dropout(0.5)
        if opt.model_type == 'Bert_p':
            output = 2
        else:
            output = opt.clabel_nb
        self.classifier = nn.Linear(hidden, output)
    def forward(self, text):
        # (bs, 768 or 1024)
        text = self.bert(text)[1]
        #text = self.dropout(text)
        text = self.classifier(text)
        return text

