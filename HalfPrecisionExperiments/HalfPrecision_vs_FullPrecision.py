

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
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

random_state = 43

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
          
          if model_choice == "roberta-large":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 384
            self.encoderModel = model_encoding

          elif model_choice == "microsoft/deberta-v2-xlarge":

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1536
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
            for param in self.encoderModel.embeddings.parameters():
                param.requires_grad = False





          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(ids, 
                   						   attention_mask=mask)

          sequence_output = total_output['last_hidden_state']

          linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size))
          linear2_output = self.linear2(linear1_output)

          return linear2_output



############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
dataset = 'sci-cite'

chosen_checkpoint = "paper_results_text_classification/distilbert-base-uncased/sci-cite/5e-06_2_True_10False_Run_0.pt"

half_configuration = True

model_choice = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

############################################################

num_epochs = 100 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
#chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 12 #12 layers for BERT total, 24 layers for T5 and RoBERTa, 48 for DeBERTa XXL
frozen_embeddings = True
average_hidden_state = False

validation_set_scoring = True
assigned_batch_size = 8
gradient_accumulation_multiplier = 4

num_warmup_steps = 100

learning_rate_choices = [1e-4, 2e-4, 1e-5, 2e-5, 5e-5, 5e-6]

mlp_classifier = False


############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

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


train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

print("Number of labels: " + str(len(set(train_set_label))))

############################################################

model = CustomBERTModel(len(set(train_set_label)), model_choice, current_dropout, 
                        frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)

model.to(device)

print("Loading the Best Model")

model.load_state_dict(torch.load(chosen_checkpoint))

if half_configuration == True:
	print("Using Half Precision!")
	model.half()

############################################################


print("Beginning Evaluation")

metric = load_metric("accuracy")
#model.eval()

total_predictions = torch.FloatTensor([]).to(device)
total_references = torch.FloatTensor([]).to(device)

inference_start = time.time()

progress_bar = tqdm(range(len(eval_dataloader)))
for batch in eval_dataloader:

    with torch.no_grad():

        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

        outputs = model(**new_batch)

        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

        total_predictions = torch.cat((total_predictions, predictions), 0)
        total_references = torch.cat((total_references, batch['labels'].to(device)), 0)

        progress_bar.update(1)


############################################################

print("--------------------------")
print("Predictions Shapes")
print(total_predictions.shape)
print(total_references.shape)

results = metric.compute(references=total_references, predictions=total_predictions)
print("Accuracy for Test Set: " + str(results['accuracy']))

f_1_metric = load_metric("f1")
macro_f_1_results = f_1_metric.compute(average='macro', references=total_references, predictions=total_predictions)
print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
micro_f_1_results = f_1_metric.compute(average='micro', references=total_references, predictions=total_predictions)
print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))



