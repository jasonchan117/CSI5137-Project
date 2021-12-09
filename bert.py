import csv
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM


__author__ = "Heng Zhang"
__email__  = "hzhan274@uOttawa.ca"
# do check your pip has installed transformers and torch


def test(csv_name):
    ### Read data from csv
    csv_content = []
    with open (csv_name, 'r', encoding="utf8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            csv_content.append(row[0].split(";")[2]) ### Customization Required
    so, po = do_bert(csv_content[1])
    print (so, po)

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


def do_bert_in_arr(text_corpus):
    '''
    Use BERT model to convert every text in the corpus into a verctor with reference to labels (line 21 - 36)
    @param: text_corpus (str arr)
    @output:  arr of torch.Tensor (sequence_output), arr of torch.Tensor (pooled_output)
    '''
    sequence_output = []
    pooled_output = []
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
    sequence_classification_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased-finetuned-mrpc')
    sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')
    for text in text_corpus:
        for label in labels:
            # Get encoding info for each text in corpus along with every labels
            encodes = sequence_classification_tokenizer.encode_plus(label, text)

            # Get encoding info from BERT
            tokens_tensor = torch.tensor([encodes['input_ids']])
            segments_tensors = torch.tensor(encodes['token_type_ids'])
            bert_model.eval()
            outputs = bert_model(tokens_tensor, token_type_ids = segments_tensors)
            sequence_output.append(outputs[0])
            pooled_output.append(outputs[1])
    return sequence_output, pooled_output

def do_bert(text):
    '''
    Use BERT model to convert text into a verctor with reference to labels (line 21 - 36)
    @param: text (str)
    @output:  torch.Tensor (sequence_output), torch.Tensor (pooled_output)
    '''
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
    sequence_classification_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased-finetuned-mrpc')
    sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')
    for label in labels:
        # Get encoding info for each text in corpus along with every labels
        encodes = sequence_classification_tokenizer.encode_plus(text)
        # Get encoding info from BERT
        tokens_tensor = torch.tensor([encodes['input_ids']])
        segments_tensors = torch.tensor(encodes['token_type_ids'])
        bert_model.eval()
        outputs = bert_model(tokens_tensor, token_type_ids = segments_tensors)
    return outputs[0], outputs[1]

def main():
    test("promise_nfr.csv")

if __name__ == "__main__":
    main()