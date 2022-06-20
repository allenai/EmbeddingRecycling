

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
import copy

from sklearn.model_selection import train_test_split

############################################################

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoderModel.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep, len(oldModuleList)):
        print(i)
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoderModel.encoder.layer = newModuleList

    return copyOfModel

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings):

          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice == "t5-3b":

            model_encoding = T5EncoderModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "roberta-large":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 768
            self.encoderModel = model_encoding



          if frozen == True:
            print("Freezing the model parameters")
            for param in self.encoderModel.parameters():
                param.requires_grad = False



          if frozen_layer_count > 0:

            if model_choice == "t5-3b":

                print("Freezing T5-3b")
                print("Number of Layers: " + str(len(self.encoderModel.encoder.block)))

                for parameter in self.encoderModel.parameters():
                    parameter.requires_grad = False

                for i, m in enumerate(self.encoderModel.encoder.block):        
                    #Only un-freeze the last n transformer blocks
                    if i+1 > 24 - frozen_layer_count:
                        print(str(i) + " Layer")
                        for parameter in m.parameters():
                            parameter.requires_grad = True

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]

                print("Length of Frozen Layers: " + str(len(layers_to_freeze)))

                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False



          
          if frozen_embeddings == True:
            print("Frozen Embeddings Layer")
            for param in self.encoderModel.embeddings.parameters():
                param.requires_grad = False





          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, ids, mask, inputs_embeds):
          
                total_output = self.encoderModel.encoder(inputs_embeds)

                sequence_output = total_output['last_hidden_state']
                linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size))
                linear2_output = self.linear2(linear1_output)

                return linear2_output



############################################################

device = "cuda:0"
device = torch.device(device)


classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']

num_epochs = 1 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
chosen_learning_rate = 5e-6 #5e-7, 5e-6, 1e-5, 2e-5, 5e-5, 0.001, 0.0001
frozen_layers = 12 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False
validation_set_scoring = False

switch_to_embeddings = False

model_choice = 'roberta-large'
assigned_batch_size = 4
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)


############################################################

def tokenize_function(examples):

    current_embeddings = tokenizer(examples["text"], padding="max_length", truncation=True)

    return current_embeddings

############################################################

for dataset in classification_datasets:

    checkpoint_path = '__checkpoint_Precomputed_' + dataset  + '_' + model_choice + '.pt'

    print("GPU Memory available at the start")
    print(get_gpu_memory())

    #print("Actual memory usage")
    #from pynvml import *
    #nvmlInit()
    #h = nvmlDeviceGetHandleByIndex(0)
    #info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total}')
    #print(f'free     : {info.free}')
    #print(f'used     : {info.used}')

    execution_start = time.time()

    print("Dataset: " + dataset)
    print("Model: " + model_choice)
    print("Dropout: " + str(current_dropout))
    print("Frozen Choice: " + str(frozen_choice))
    print("Number of Runs: " + str(number_of_runs))
    print('Learning Rate: ' + str(chosen_learning_rate))
    print("Checkpoint Path: " + checkpoint_path)
    print("Number of Frozen Layers: " + str(frozen_layers))
    print("Frozen Embeddings: " + str(frozen_embeddings))
    print("Patience: " + str(patience_value))
    print("Average Hidden Layers: " + str(average_hidden_state))
    print("Validation Set Choice: " + str(validation_set_scoring))
    print("Number of Epochs: " + str(num_epochs))

    














    ############################################################


    print("Beginning training using embeddings")

    #print(eval_embeddings.shape)

    train_embeddings = torch.load('training_embeddings.pt')
    train_labels = torch.load('training_labels.pt')

    eval_dataloader = DataLoader(train_embeddings, batch_size=assigned_batch_size)
    eval_dataloader_labels = DataLoader(train_labels, batch_size=assigned_batch_size)

    switch_to_embeddings = True


    ############################################################

    model = CustomBERTModel(13, model_choice, current_dropout, 
                                frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(device)


    new_model = deleteEncodingLayers(model, 12)
    print("Number of Layers of New Model: " + str(len(new_model.encoderModel.encoder.layer)))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6
    #optimizer = Adam(model.parameters(), lr=1e-5) #5e-6

    num_training_steps = num_epochs * len(eval_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    ############################################################

    metric = load_metric("accuracy")

    total_predictions = torch.FloatTensor([]).to(device)
    total_references = torch.FloatTensor([]).to(device)

    inference_start = time.time()

    count = 0

    progress_bar = tqdm(range(len(eval_dataloader)))

    new_model.train()
    for batch, labels in zip(eval_dataloader, eval_dataloader_labels):

        #with torch.no_grad():

                new_batch = {'ids': torch.FloatTensor([]).to(device), 
                			 'mask': torch.FloatTensor([]).to(device), 
                			 'inputs_embeds': batch.to(device)}

                outputs = new_model(**new_batch)

                loss = criterion(outputs, labels.long().to(device))

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)



    inference_end = time.time()
    total_inference_time = inference_end - inference_start

    ############################################################

    print("--------------------------")
    print("Predictions Shapes")
    print(total_predictions.shape)
    print(total_references.shape)

    results = metric.compute(references=total_predictions, predictions=total_references)
    print("Accuracy for Test Set: " + str(results['accuracy']))

    print("Training Time using Precomputed Embeddings: " + str(total_inference_time))

