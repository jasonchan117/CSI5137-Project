import csv
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from transformers import BertTokenizer, BertModel, BertForMaskedLM

class Dataset(data.Dataset):

    # A dictionary that stores the descriptions of labels (shorter version about 20 words each)
    labels_1 = {
        "F": "Description of the service that the software must offer. It describes a software or its component.",
        "US": "Degree to which a software can achieve quantified objectives with effectiveness, efficiency, and satisfaction in a quantified context of use",
        "SE": "Protection of software from information disclosure, theft of or damage to their data, also from the disruption or misdirection of its services",
        "O": "Ability to keep the software in a safe and reliable functioning condition, according to pre-defined operational requirements.",
        "PE": "Amount of useful work accomplished by a software, estimated in terms of accuracy, efficiency and speed of executing",
        "LF": "Aspects of software's design, including elements, the behavior of elements, non-graphical user interface, and API not related to functional properties",
        "A"  : "Description of shoftware's accessibility for a user at a given point in time. Expressed as a probability percentage",
        "SC": "Property of software to handle increasing (or decreasing) amount of workload by adding resources to the software",
        "MN": "Ease with which a software can be maintained for learning from the past in order to improve the ability or reliability of software",
        "L": "Requirement that limits the risk for legal disputes between a software provider and their users, and the compliance with laws",
        "FT" : "Property that enables software to continue operating properly in the event of the failure of faults within its components",
        "PO" : "Usability of the same software in different environments, also the key issue for development cost reduction",
    }
    # A dictionary that stores the descriptions of labels (longer version about 44 words each)
    labels_2 = {
        # US, SE, O, PE
        "F": "It defines a function of a software or its component, where a function is described as a specification of behavior between inputs and outputs; it may involve calculations, technical details, data manipulation, processing, and other specific functionality that define what is supposed to accomplish",
        "US": "Degree to which a software can be used by specified consumers to achieve quantified objectives with effectiveness, efficiency, and satisfaction in a quantified context of use; it considers user satisfaction and utility and aims to improve user experience through iterative design",
        "SE": "Protection of software from information disclosure, theft of or damage to its electronic data, and from the disruption or misdirection of the services it provides; it specifies the security properties, security constraints, or security practice that the software must possess",
        "O": "Ability to keep a function, a system, or the whole software in a safe and reliable functioning condition, according to pre-defined operational requirements; this includes the ability of software, systems, and business processes to work together to accomplish a common task.",
        "PE": "Amount of useful work accomplished by a software, estimated in terms of accuracy, efficiency and speed of executing computer program instructions; it is tested to determine how a software performs in terms of responsiveness and stability under a particular workload",
        "LF": "Aspects of software's design, including elements like colors, shapes, layout, and typefaces, and the behavior of dynamic elements like buttons, boxes, and menus; can also refer to aspects of a non-graphical user interface or an API that are not related to its functional properties",
        "A"  : "Description of how likely the software is accessible for a user at a given point in time. It can be expressed as a probability percentage, may also be defined as a percentage of time the system is accessible for operation during some time period.",
        "SC": "Property of a software to handle a growing amount of work by adding resources to the software; a software is expected to handle extra works when it has been given greater hardware resources and to preserve existing activities when the available hardware resources decrease",
        "MN": "Ease with which a software can be modified after delivery to correct faults to improve the ability to maintain software, or improve the reliability of software based on maintenance experience; a common perception of maintenance is that it merely involves fixing defects",
        "L": "Requirement that limits the risk for legal disputes between a software provider and their users; it involves determining the applicable regulations, extracting requirements and other key concepts, creating the policies necessary to achieve compliance with those regulations, and monitoring compliance throughout the software lifecycle",
        "FT" : "Property that enables a software to continue operating properly in the event of the failure of one or more faults within some of its components, can be achieved by anticipating exceptional conditions and building the system to cope with errors",
        "PO" : "Usability of the same software in different environments; the prerequirement for portability is the generalized abstraction between the application logic and system interfaces, and when software with the same functionality is produced for several computing platforms, portability is the key issue for development cost reduction"

    }


    def __init__(self, opt):
        '''
        attributes:
        opt            :   parser specified in train.py
        text_corpus    :   text corpus, string array
        text_vectors   :   ***vectors that represent text in corpus, 长度为num_hidden_layers的(batch_size， sequence_length，hidden_size)的Tensor.列表
        p_labels       :   ***parent labels text in corpus, array of bitmaps (2d array)    [NFR, F]
        c_labels       :   ***child labels text in corpus, array of bitmaps (2d array)     [F, A, FT, L, LF, MN, O, PE, PO, SC, SE, US]
        input_ids      :   arraies of words' (in text) ids in dictionary, array of tensors

        sequence_classification_tokenizer is used internally only for tokenization
        '''
        self.opt = opt
        self.text_corpus = []
        self.input_ids = []
        self.p_labels = []
        self.c_labels = []
        self.labels = [self.labels_1, self.labels_2]
        self.sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                                                           'bert-base-cased-finetuned-mrpc')

        df = pd.read_csv(self.opt.dataset, delimiter=';', header=0, encoding='utf8',
                         names=['number', 'ProjectID', 'RequirementText', 'class', 'NFR', 'F', 'A', 'FT', 'L', 'LF',
                                'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US'])
        df = df.dropna()
        # df = df.sample(frac=1, axis=0, random_state=904727489)
        self.text_corpus = df.RequirementText.tolist()
        labels = ["NFR","F","US","SE","O","PE","LF","A","SC","MN","L","FT","PO"]
        self.label_names = labels
        # indice = []
        for index, row in df.iterrows():
            temp_p = []
            temp_c = []
            for i in labels[0:2]:
                temp_p.append(float(row[i]))
            for i in labels[1:opt.clabel_nb + 1]:
                temp_c.append(float(row[i]))
            self.p_labels.append(temp_p)
            if self.opt.clabel_nb != 12:
                temp_c.append(0.)
                if temp_c == [0.] * (self.opt.clabel_nb + 1):
                    temp_c[-1] = 1.
            # If the task is doing the sublabel classification of NFR and it doesn't contain all the sublabels, we need to consider one more class, which is (OTH)other.
            self.c_labels.append(temp_c)

        for ind, text in enumerate(self.text_corpus):
            # Get encoding info for each text in corpus along with every labels
            # if self.opt.clabel_nb != 5:
            encodes = self.sequence_classification_tokenizer.encode_plus(text)

            tokens_tensor = encodes['input_ids']
            self.input_ids.append(tokens_tensor)

        self.input_ids = np.array(self.input_ids)
        self.p_labels = np.array(self.p_labels)
        self.c_labels = np.array(self.c_labels)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: input_ids, parent_label, child_label, and BERT vector of a data instance at specified location
                   ^ this is the array of words' id in dictionary
        '''
        #Get quantity of children labels are desired in classification

        input_ids = self.input_ids[index]
        parent_label = self.p_labels[index]
        child_label = self.c_labels[index]


        if len(input_ids) < self.opt.sen_len:
            input_ids = np.append(input_ids, [0 for i in range(self.opt.sen_len - len(input_ids))])
        else:
            input_ids[self.opt.sen_len - 1] = 102
        return torch.tensor(input_ids[0:self.opt.sen_len]), torch.tensor(parent_label), torch.tensor(child_label)


    def __len__(self):
        '''
        @return: the shape of vector, tuple (quantity of data, length of vectors)
        '''
        return len(self.input_ids)



    def getLabelDes(self):
        '''
        @return array of tensor (representing label descriptions based on their tokens' ids in the dictionary)
        '''
        #Get quantity of children labels are desired in classification
        clabel_nb = self.opt.clabel_nb

        res = []
        counter = 0
        max_d = -1

        for label in self.labels[self.opt.des_ver-1]:
            if counter == clabel_nb:
                break
            counter += 1
            des = self.sequence_classification_tokenizer.encode_plus(self.labels[self.opt.des_ver-1][label])['input_ids']

            max_d = max(max_d, len(des))
        counter = 0
        for label in self.labels[self.opt.des_ver-1]:
            if counter == clabel_nb:
                break
            counter += 1
            des = self.sequence_classification_tokenizer.encode_plus(self.labels[self.opt.des_ver-1][label])['input_ids']

            if len(des) < max_d:
                des = np.append(des, [0 for i in range(max_d - len(des))])
            res.append(des)
        res = res[:self.opt.clabel_nb]


        return torch.tensor(res)

'''
For Testing Purpose:

def main():

if __name__ == "__main__":
    main()
'''