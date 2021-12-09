import csv
import torch
import numpy as np
import torch.utils.data as data
from transformers import BertTokenizer, BertModel, BertForMaskedLM

class Dataset(data.Dataset):

    # A dictionary that stores the descriptions of labels, order of entires is used to create bitmaps
    labels = {
        "F"  : "A description of the service that the software must offer. It describes a software system or its component.",
    }
    '''
        "A"  : "",
        "FT" : "",
        "L"  : "",
        "LF" : "",
        "MN" : "",
        "O"  : "",
        "PE" : "",
        "PO" : "",
        "SC" : "",
        "SE" : "",
        "US" : "",
    '''

    def __init__(self):
        '''
        attributes:
        opt            :   parser specified in train.py  !!! DELETED !!! since train.py imports current file, current file cannot import things from train.py
        text_corpus    :   text corpus, string array
        text_vectors   :   ***vectors that represent text in corpus, 长度为num_hidden_layers的(batch_size， sequence_length，hidden_size)的Tensor.列表
        p_labels       :   ***parent labels text in corpus, array of bitmaps (2d array)    [NFR, F]
        c_labels       :   ***child labels text in corpus, array of bitmaps (2d array)     [F, A, FT, L, LF, MN, O, PE, PO, SC, SE, US]
        input_ids      :   arraies of words' (in text) ids in dictionary, array of tensors

        sequence_classification_tokenizer is used internally only for tokenization
        '''
        self.text_corpus = []
        self.text_vectors = []
        self.input_ids = []
        self.token_type_ids = []
        self.p_labels = []
        self.c_labels = []

        ### Read data from csv
        with open ("promise_nfr.csv", 'r', encoding="utf8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader) ### Skip Heading Row
            for row in csv_reader:
                a_row = row[0].split(";")
                self.text_corpus.append(a_row[2])
                self.p_labels.append(a_row[4:6])
                self.c_labels.append(a_row[5:])

        ### Create BERT Model
        bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')

        ### Calculate BERT vectors
        for text in self.text_corpus:
            # Get encoding info for each text in corpus along with every labels
            encodes = self.sequence_classification_tokenizer.encode_plus(text)
            # Get encoding info from BERT
            tokens_tensor = torch.tensor([encodes['input_ids']])
            self.input_ids.append(tokens_tensor)
            bert_model.eval()
            outputs = bert_model(tokens_tensor)
            self.text_vectors.append(outputs[0])



    def __getitem__(self, index, clabel_nb = 12):
        '''
        @param: index: int, location of data instance
        @param: clabel_nb int, how many children labels are required in classification, default value = 12
        @return: input_ids, parent_label, child_label, and BERT vector of a data instance at specified location
                   ^ this is the array of words' id in dictionary
        '''
        input_ids = self.input_ids[index]
        parent_label = self.p_labels[index]
        child_label = self.c_labels[index][0:clabel_nb]
        bert_vector = self.text_vectors[index]
        return input_ids, parent_label, child_label, bert_vector




    def __len__(self):
        '''
        @return: the shape of vector, tuple (quantity of data, length of vectors)
        '''
        return (len(self.text_vectors), self.text_vectors[0].shape[2]) 



    def getLabelDes(self, clabel_nb = 12):
        '''
        @return array of tensor (representing label descriptions based on their tokens' ids in the dictionary)
        '''
        res = []
        counter = 0
        for label in self.labels:
            if counter == clabel_nb:
                break
            counter ++
            res.append(torch.tensor(self.sequence_classification_tokenizer.encode_plus(self.labels[label])['input_ids']))
        return res

'''
For Testing Purpose:
'''
def main():
    corpus = Dataset()
    print(corpus[10])
    print(corpus.__len__())
    print(corpus.getLabelDes())

if __name__ == "__main__":
    main()