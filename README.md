python modules required:

torch transformers importlib-metadata 
tqdm boto3 requests regex sentencepiece sacremoses


How to run this project:
go to current directory, and execute following in command/terminal:
    *python train.py --id [a customized name/id goes here]*
or run this project on GPU:
    *python train.py --id [a customized name/id goes here] --id cuda True*

A folder is needed to store the model's computational data, name it by ckpt 
OR something else you like by adding one more parameters in command:
    *python train.py --id [a customized training id goes here] --ckpt [a customized folder name goes here]*


There is a severe data imbalance (for child label) in dataset, and some labels' instances have become noises 
due to extremely low quantity, so we can only use the child labels that have sufficient data instances used 
in classification to let our model be more adaptive to the main stream labels:
    *python train.py --id [a customized training id goes here] --clabel_nb 5*