
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

from sklearn.model_selection import train_test_split

############################################################

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings):

          super(CustomBERTModel, self).__init__()

          if model_choice == 'roberta-large':

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
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

          
          if frozen_embeddings == True:
            print("Frozen Embeddings Layer")
            for param in self.encoderModel.embeddings.parameters():
                param.requires_grad = False



          ##################################################################

          self.roberta_mlp = nn.Sequential(
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 768)
                             )

          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, roberta_ids, roberta_mask):

          roberta_output = finetuned_roberta_model(roberta_ids, attention_mask=roberta_mask)
          roberta_output = roberta_output['last_hidden_state']
          roberta_output_reduced = self.roberta_mlp(roberta_output)
          
          #total_output = self.encoderModel(input_embeds=roberta_output)
          total_output = self.encoderModel.encoder(roberta_output_reduced)

          scibert_output = total_output['last_hidden_state']
          scibert_output = scibert_output[:,0,:].view(-1, self.embedding_size)

          linear1_output = self.linear1(scibert_output)
          linear2_output = self.linear2(linear1_output)

          return linear2_output



############################################################

#classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction', 'mag']
#label_count = [13, 3, 7, 7]

classification_datasets = ['mag']
label_count = [7]

num_epochs = 15 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001, 0.0001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

for dataset, labels in zip(classification_datasets, label_count):

	finetuned_roberta_model = CustomBERTModel(labels, 'roberta-large', current_dropout, 
											  frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)
	roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large', model_max_length=512)

	finetuned_roberta_path = "pretrained_roberta-large_" + dataset + "_for_Scibert_mapping.pt"
	finetuned_roberta_model.load_state_dict(torch.load(finetuned_roberta_path), strict=False)

	finetuned_roberta_model = finetuned_roberta_model.encoderModel

	new_path = "./prefinetuned_RoBERTa/new_pretrained_roberta-large_" + dataset + "_for_Scibert_mapping.pt"

	torch.save(finetuned_roberta_model.state_dict(), new_path)





