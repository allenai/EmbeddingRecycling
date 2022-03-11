import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer

import pandas as pd
import numpy as np
import ast
import datasets
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import pyarrow as pa
import pyarrow.dataset as ds

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

import torch
from tqdm.auto import tqdm


from sklearn import linear_model
import numpy as np
import json

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import f1_score

############################################################

print("With 0.5 dropout for frozen embeddings")

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, encoder_model, embedding_size, dropout_layer):
          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          self.encoderModel = encoder_model
          ### New layers:

          self.lstm = nn.LSTM(embedding_size, 200, batch_first=True,bidirectional=True, num_layers=2)


          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(
               ids, 
               attention_mask=mask)

          sequence_output = total_output['last_hidden_state']

          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings

          hidden = torch.cat((lstm_output[:,-1, :],lstm_output[:,0, :]),dim=-1)

          return hidden


############################################################

device = "cuda:0"

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']

#model_choice = "t5-3b"
#tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)
#model_encoding = T5EncoderModel.from_pretrained(model_choice)
#embedding_size = 1024
#current_dropout = False

#model_choice = 'bert-base-uncased'
#tokenizer = AutoTokenizer.from_pretrained(model_choice)
#model_encoding = BertModel.from_pretrained(model_choice)
#embedding_size = 768
#current_dropout = False
#for param in model_encoding.parameters():
#    param.requires_grad = False

model_choice = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)
model_encoding = AutoModel.from_pretrained(model_choice)
embedding_size = 768
current_dropout = False
for param in model_encoding.parameters():
    param.requires_grad = False

#model_choice = 'hivemind/gpt-j-6B-8bit'
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 4096
#current_dropout = False



############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################


for dataset in classification_datasets:

    print("Processing " + dataset + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout")

    # Chemprot train, dev, and test
    with open('text_classification/' + dataset + '/train.txt') as f:

        train_set = f.readlines()
        train_set = [ast.literal_eval(line) for line in train_set]
        train_set_text = [line['text'] for line in train_set]
        train_set_label = [line['label'] for line in train_set]

    with open('text_classification/' + dataset + '/dev.txt') as f:
        
        dev_set = f.readlines()
        dev_set = [ast.literal_eval(line) for line in dev_set]
        dev_set_text = [line['text'] for line in dev_set]
        dev_set_label = [line['label'] for line in dev_set]

    with open('text_classification/' + dataset + '/test.txt') as f:
        
        test_set = f.readlines()
        test_set = [ast.literal_eval(line) for line in test_set]
        test_set_text = [line['text'] for line in test_set]
        test_set_label = [line['label'] for line in test_set]


    ############################################################

    labels_list = sorted(list(set(train_set_label)))

    label_to_value_dict = {}

    count = 0
    for label in labels_list:
      label_to_value_dict[label] = count
      count += 1

    train_set_label = [label_to_value_dict[label] for label in train_set_label]
    dev_set_label = [label_to_value_dict[label] for label in dev_set_label]
    test_set_label = [label_to_value_dict[label] for label in test_set_label]

    ############################################################

    training_dataset_pandas = pd.DataFrame({'label': train_set_label + dev_set_label, 'text': train_set_text + dev_set_text})#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


    joint_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 'test' : test_dataset_arrow})
    tokenized_datasets = joint_dataset.map(tokenize_function, batched=True)


    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")


    ############################################################

    print("Loading Model")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8)

    print("Number of labels: " + str(len(set(train_set_label))))

    model = CustomBERTModel(len(set(train_set_label)), model_encoding, embedding_size, current_dropout)

    device = torch.device("cuda:0")
    model.to(device)

    ############################################################

    print("Gathering Traing Set")

    progress_bar = tqdm(range(len(train_dataloader)))

    total_training_set = torch.FloatTensor([]).to(device)
    total_training_labels = torch.FloatTensor([]).to(device) 

    for batch in train_dataloader:

            with torch.no_grad():
            
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels']

                #print("Batch")
                #print(batch)
                
                new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
                outputs = model(**new_batch)

                #print("outputs")
                #print(type(outputs))
                #print((outputs.shape))
                #print(outputs[0])

                total_training_set = torch.cat((total_training_set, outputs), 0)
                total_training_labels = torch.cat((total_training_labels, labels), 0)

                progress_bar.update(1)


    total_training_set = total_training_set.to('cpu').numpy()
    total_training_labels = total_training_labels.to('cpu').numpy()

    print(total_training_set.shape)
    print(total_training_labels.shape)



    ############################################################

    print("Gather Testing Set")

    metric = load_metric("accuracy")
    model.eval()

    total_testing_set = torch.FloatTensor([]).to(device)
    total_testing_labels = torch.FloatTensor([]).to(device)

    progress_bar = tqdm(range(len(train_dataloader)))

    for batch in eval_dataloader:

        with torch.no_grad():

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

            outputs = model(**new_batch)

            total_testing_set = torch.cat((total_testing_set, outputs), 0)
            total_testing_labels = torch.cat((total_testing_labels, labels), 0)

            progress_bar.update(1)



    total_testing_set = total_testing_set.to('cpu').numpy()
    total_testing_labels = total_testing_labels.to('cpu').numpy()

    print(total_testing_set.shape)
    print(total_testing_labels.shape)


    ############################################################


    print("Train Model")

    linearModel = make_pipeline(StandardScaler(),
                    SGDClassifier(loss="perceptron", learning_rate="optimal", max_iter=1000, tol=1e-3, random_state=0))

    linearModel.fit(total_training_set, total_training_labels)


    ############################################################

    print("Testing Model")

    score = linearModel.score(total_testing_set, total_testing_labels)


    print("Performance for " + dataset + " dataset")
    print(score)
    print("Total Possible Labels")
    print(len(labels_list))

    predictions = linearModel.predict(total_testing_set)
    print("F-1 Score for " + dataset + " dataset")
    print(f1_score(total_testing_labels, predictions, average='macro'))









