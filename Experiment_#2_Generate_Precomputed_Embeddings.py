


import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
import tensorflow as tf

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

import subprocess as sp
import os

############################################################

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice):

          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice == "t5-3b":

            model_encoding = T5EncoderModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "roberta-large":

            model_encoding = BertModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding



          








          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(
                ids, 
                attention_mask=mask)

          inner_hidden_state = total_output['last_hidden_state']

          return inner_hidden_state



############################################################

device = "cuda:0"
device = torch.device(device)

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['sciie-relation-extraction']





checkpoint_path = 'checkpoint42.pt' # 42, 43, 44, 45, 46, 47, 48, 49
model_choice = 'roberta-large'
assigned_batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)
first_phase_encoder = AutoModel.from_pretrained(model_choice, output_hidden_states=True)




############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids


############################################################

for dataset in classification_datasets:

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


    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")


    ############################################################



    print("Loading Model")

    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
    validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

    print("Number of labels: " + str(len(set(train_set_label))))

    ############################################################

    model = CustomBERTModel(len(set(train_set_label)), model_choice)

    model.to(device)

    ############################################################

    print("Beginning Training")

    progress_bar = tqdm(range(len(train_dataloader)))

    training_embeddings = torch.FloatTensor([])

    for batch in train_dataloader:

        with torch.no_grad():
        
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
            embeddings = model(**new_batch)

            training_embeddings = torch.cat((training_embeddings, embeddings.to('cpu')), 0)

            progress_bar.update(1)

    print("Training Embeddings Shape")
    print(training_embeddings.shape)

    torch.save(training_embeddings, 'Experiment2_Tensors/' + dataset + '_' + model_choice + '_training.pt')


    #############################################################

    validation_embeddings = torch.FloatTensor([])

    progress_bar = tqdm(range(len(validation_dataloader)))

    for batch in validation_dataloader:

        with torch.no_grad():
        
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
            embeddings = model(**new_batch)

            validation_embeddings = torch.cat((validation_embeddings, embeddings.to('cpu')), 0)

            progress_bar.update(1)


    torch.save(validation_embeddings, 'Experiment2_Tensors/' + dataset + '_' + model_choice + '_validation.pt')

    ############################################################

    testing_embeddings = torch.FloatTensor([])

    progress_bar = tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:

        with torch.no_grad():

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

            embeddings = model(**new_batch)

            testing_embeddings = torch.cat((testing_embeddings, embeddings.to('cpu')), 0)

            progress_bar.update(1)



    torch.save(testing_embeddings, 'Experiment2_Tensors/' + dataset + '_' + model_choice + '_testing.pt')




