

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer

import pandas as pd
import numpy as np
import ast
import datasets
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import pyarrow as pa
import pyarrow.dataset as ds

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_scheduler

import torch
from tqdm.auto import tqdm
import statistics
import time

############################################################

print("New checkpoint path, fixed yet again")

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, dropout_layer, frozen, relu):
          super(CustomBERTModel, self).__init__()

          if model_choice == "t5-3b":

            model_encoding = T5EncoderModel.from_pretrained(model_choice)
            self.embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "roberta-large":

            model_encoding = AutoModel.from_pretrained(model_choice)
            self.embedding_size = 1024
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice)
            self.embedding_size = 768
            self.encoderModel = model_encoding



          if frozen == True:
            for param in self.encoderModel.parameters():
                param.requires_grad = False


          ### New layers:

          #self.lstm = nn.LSTM(embedding_size, 256, batch_first=True,bidirectional=True, num_layers=2)
          #self.linear = nn.Linear(256*2, number_of_labels)


          self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=200, batch_first=True, 
                              bidirectional=True, num_layers=2, dropout=0.5)

          if relu == True:

            self.perceptron = nn.Sequential(
                          nn.Linear(200*2, 200),
                          nn.ReLU(),
                          nn.Linear(200, number_of_labels)
                          )

          else:

              self.perceptron = nn.Sequential(
                              nn.Linear(200*2, 200),
                              #nn.ReLU(),
                              nn.Linear(200, number_of_labels)


                              #nn.ReLU(),
                              #nn.Linear(400, 200),
                              #nn.ReLU(),
                              #nn.Linear(200, 200),
                              #nn.Dropout(p=0.5),
                              #nn.Linear(200, 100),
                              #nn.ReLU(),
                              #nn.Linear(200, number_of_labels)
                              #nn.Linear(200, number_of_labels)
                        )

          self.dropout_layer = dropout_layer


          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(
               ids, 
               attention_mask=mask)

          sequence_output = total_output['last_hidden_state']
          #pooler_output = total_output['pooler_output']

          #print('pooler_output')
          #print(type(pooler_output))
          #print(pooler_output.shape)

          if self.dropout_layer == True:
              dropout_layer = nn.Dropout(p=0.5)
              sequence_output = dropout_layer(sequence_output)

          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings

          #hidden = torch.cat((lstm_output[:,0, :],lstm_output[:,-1, :]),dim=-1)
          hidden = torch.cat((lstm_output[:,-1, :200],lstm_output[:,0, 200:]),dim=-1)
          #hidden = torch.cat((lstm_output[:,-1, :],lstm_output[:,0, :]),dim=-1)

          #print('hidden')
          #print(hidden.shape)

          linear_output = self.perceptron(hidden)

          #print('linear_output')
          #print(linear_output.shape)

          #hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)

          #if self.dropout_layer == True:
          #  print("Performing dropout")
          #  dropout_layer = nn.Dropout(p=0.5)
          #  hidden = dropout_layer(hidden)
          
          #linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification

          return linear_output


############################################################

device = "cuda:0"

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']

num_epochs = 100 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
current_relu = True
number_of_runs = 1 #1 #5
frozen_choice = False
chosen_learning_rate = 5e-5 #5e-6, 1e-5, 2e-5, 5e-5, 0.001


#checkpoint_path = 'checkpoint51.pt'
#model_choice = "t5-3b"
#assigned_batch_size = 2
#tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)

#checkpoint_path = 'checkpoint61.pt'
#model_choice = 'bert-base-uncased'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

checkpoint_path = 'checkpoint71.pt'
model_choice = 't5-3b'
assigned_batch_size = 32
tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)
                                             #attention_probs_dropout_prob=0.5)
                                                      #hidden_dropout_prob=0.5)


#checkpoint_path = 'checkpoint84.pt'
#model_choice = 'roberta-large'
#assigned_batch_size = 1
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)


#model_choice = 'hivemind/gpt-j-6B-8bit'
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 4096



############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"])#.input_ids

############################################################

execution_start = time.time()

for dataset in classification_datasets:

    print("Dataset: " + dataset)
    print("Model: " + model_choice)
    print("Relu: " + str(current_relu))
    print("Dropout: " + str(current_dropout))
    print("Frozen Choice: " + str(frozen_choice))
    print("Number of Runs: " + str(number_of_runs))
    print('Learning Rate: ' + str(chosen_learning_rate))
    print("Number of Epochs: " + str(num_epochs))
    print("Checkpoint Path: " + checkpoint_path)

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

    training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

    validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:1000]
    validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
    validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


    classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                    'validation': validation_dataset_arrow, 
                                    'test' : test_dataset_arrow})
    tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)


    print(max([len(tokens) for tokens in tokenized_datasets['train']['input_ids']]))
    print(max([len(tokens) for tokens in tokenized_datasets['validation']['input_ids']]))
    print(max([len(tokens) for tokens in tokenized_datasets['test']['input_ids']]))


    #print("tokenized_datasets")
    #print(tokenized_datasets['train']['input_ids'])




