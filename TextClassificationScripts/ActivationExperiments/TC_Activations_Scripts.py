

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
#import tensorflow as tf

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
import json
import random

#############################################################

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

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
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice == "t5-3b":

            model_encoding = T5EncoderModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "roberta-large" or model_choice == "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large" or model_choice == "microsoft/deberta-v3-xsmall":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 384
            self.encoderModel = model_encoding

          elif model_choice == "t5-small":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 512
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice)
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

            elif model_choice == "distilbert-base-uncased":

                #print(self.encoderModel.__dict__)
                print("Number of Layers: " + str(len(list(self.encoderModel.transformer.layer))))

                layers_to_freeze = self.encoderModel.transformer.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False



          
          if frozen_embeddings == True:
            print("Frozen Embeddings Layer")
            #print(self.encoderModel.__dict__)
            for param in self.encoderModel.embeddings.parameters():
                param.requires_grad = False





          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, ids, mask):
          
          if model_choice == "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune":

              total_output = self.encoderModel(
                   input_ids=ids,
                   decoder_input_ids=ids, 
                   attention_mask=mask)

          elif model_choice == "t5-small":

              total_output = self.encoderModel(
                   input_ids=ids,
                   decoder_input_ids=ids, 
                   attention_mask=mask)

          else:

              total_output = self.encoderModel(
                   ids, 
                   attention_mask=mask)

          sequence_output = total_output['last_hidden_state']

          if self.average_hidden_state == True:

            sequence_output = torch.mean(sequence_output, dim=1)
            linear1_output = self.linear1(sequence_output)

          else:

            linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size))


          linear2_output = self.linear2(linear1_output)

          return linear2_output




############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################

device = "cuda:0"
device = torch.device(device)

num_epochs = 100 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
number_of_runs = 10 #1 #5
frozen_choice = False
#chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = False
assigned_batch_size = 8
gradient_accumulation_multiplier = 4

############################################################

#classification_datasets = ['chemprot', 'sci-cite', "sciie-relation-extraction"]
dataset = 'chemprot'

chosen_learning_rate = 1e-5
model_choice = 'roberta-large'

tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

############################################################

# Chemprot train, dev, and test
with open('text_classification/' + dataset + '/train.txt') as f:

    train_set = f.readlines()
    train_set = [ast.literal_eval(line) for line in train_set]
    train_set_text = [line['text'] for line in train_set]
    train_set_label = [line['label'] for line in train_set]

with open('text_classification/' + dataset + '/dev.txt') as f:
    
    dev_set = f.readlines()
    dev_set = [ast.literal_eval(line) for line in dev_set]

    dev_set_text = []
    dev_set_label = []
    for line in dev_set:

        # Fix bug in MAG dev where there is a single label called "category"
        if line['label'] != 'category':
            dev_set_text.append(line['text'])
            dev_set_label.append(line['label'])
        else:
            print("Found the error with category")

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

############################################################

classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                        	   'validation': validation_dataset_arrow, 
                                        	   'test' : test_dataset_arrow})
tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

############################################################

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

############################################################












############################################################

fully_finetuned_model = CustomBERTModel(len(set(train_set_label)), model_choice, current_dropout, 
                        				frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)

fully_finetuned_checkpoint_path = "paper_results_text_classification/" + model_choice.replace("/", "-") + "/"+ dataset + "/" + str(chosen_learning_rate) + "_"
fully_finetuned_checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
fully_finetuned_checkpoint_path += str(validation_set_scoring) + "_Run_" + str(0) + ".pt"

fully_finetuned_model.load_state_dict(torch.load(fully_finetuned_checkpoint_path))

fully_finetuned_model.to(device)

############################################################

frozen_layers = 12
frozen_embeddings = True
chosen_learning_rate = 1e-05

semifrozen_checkpoint_path = "paper_results_text_classification/" + model_choice.replace("/", "-") + "/" + dataset + "/" + str(chosen_learning_rate) + "_"
semifrozen_checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
semifrozen_checkpoint_path += str(validation_set_scoring) + "_Run_" + str(0) + ".pt"

semifrozen_model = CustomBERTModel(len(set(train_set_label)), model_choice, current_dropout, 
                        		   frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)

#semifrozen_checkpoint_path = "semifrozen_roberta_chemprot_checkpoint"
semifrozen_model.load_state_dict(torch.load(semifrozen_checkpoint_path))

semifrozen_model.to(device)

############################################################


print("Beginning Evaluation")

metric = load_metric("accuracy")

total_predictions = torch.FloatTensor([]).to(device)
total_references = torch.FloatTensor([]).to(device)

inference_start = time.time()

progress_bar = tqdm(range(len(eval_dataloader)))
for batch in eval_dataloader:

    with torch.no_grad():

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']

        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

        finetuned_outputs = fully_finetuned_model(**new_batch)

        logits = finetuned_outputs
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

        total_predictions = torch.cat((total_predictions, predictions), 0)
        total_references = torch.cat((total_references, labels), 0)

        ##################################################################

        semifrozen_outputs = semifrozen_model(**new_batch)

        ##################################################################

        progress_bar.update(1)

