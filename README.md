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

