
import torch
import numpy as np
import torch.nn as nn

class F_HMN(nn.Module):
    def __init__(self, opt):
        # The bert model object
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.opt = opt
    def forward(self, text, token_type_ids, child_label_des):
        # child_label_des: (12, 18)


        # (12, 768)
        des_embed = self.bert(child_label_des, token_type_ids[0])[0]
        # (bs, 12, 768)
        des_embed = des_embed.repeat((self.opt.batchsize, 1, 1))



