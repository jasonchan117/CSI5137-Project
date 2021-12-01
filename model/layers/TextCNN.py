import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class TextCNN(nn.Module):
    def __init__(self, word_embedding_dimension, sentence_max_size, label_num):
        super(TextCNN, self).__init__()
        Dim = word_embedding_dimension
        Cla = label_num
        Ci = 2
        Knum = 2
        Ks = [4, 3, 2]
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks) * Knum, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, Cla)
    def forward(self, x):

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        logit = self.fc1(logit)
        logit = self.fc2(logit)
        logit = self.fc3(logit)
        return logit
