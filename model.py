
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class F_HMN(nn.Module):
    def __init__(self, opt):
        # The bert model object
        self.bert_description = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.bert_text = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.opt = opt
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
        cosine = cos(a,b).unsqueeze(2)
        # [B*B,L,H]*[B*B,L,1]
        attention = a*cosine
        # [B, B, L, H]
        part_list = attention.view(len(input_list), len(input_list), -1, self.args.hidden_size)
        # [B,B,L,H]----[B,1,L,H]-----[B,L,H]  all subclasses's attention
        avgpool = F.avg_pool3d(part_list, (len(input_list), 1, 1)).squeeze()

        # weight = torch.div(avgpool,input_list)

        output_list = input_list-avgpool
        father = torch.cat([avgpool])

    def forward(self, text, token_type_ids, child_label_des, child_label_len):
        # child_label_des: (12, max), child_label_len: (12, )



        des_output = self.bert_description(child_label_des, token_type_ids[0])[1]
        # (12, max, 768)
        des_per_embed = des_output[0]
        # (12, 768)
        des_embed = des_output[1]

        nf_embed = des_embed[:11] #
        f_embed = des_embed[11:]

        nf_per_embed = des_per_embed[:11]
        f_per_embed = des_per_embed[11:]

        nf_output = self.cal(nf_per_embed, nf_embed)
        f_output = self.cal(f_per_embed, f_embed)
        parent_label = torch.cat([nf_output[0], f_output[0]])
        all_list = [F.max_pool1d(nf_output[1].transpose(1, 2), nf_output[1].transpose(1, 2).size(2)).squeeze(2), F.max_pool1d(f_output[1].transpose(1, 2), f_output[1].transpose(1, 2).size(2)).squeeze(2)]
        # parent_label shape:(12, 768)
        label_des = F.max_pool1d(parent_label.transpose(1, 2), parent_label.transpose(1, 2).size(2)).squeeze(2)

        label_repeat_out = label_des.repeat((text.size(0), 1, 1))
        parent_output =
        # (bs, 18, 768)
        text_embed = self.bert_text(text, token_type_ids[0])[0]
