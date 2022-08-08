

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, RobertaForSequenceClassification
from transformers import BertModel, AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaForTokenClassification, AutoModelForTokenClassification
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

from tokenizers import PreTokenizedInputSequence
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

def process_NER_dataset(dataset_path):

    total_words = []
    total_labels = []

    current_words = []
    current_labels = []

    with open(dataset_path) as f:

        train_set = f.readlines()

        for line in tqdm(train_set):

            line_split = line.split("\t")

            if len(line_split) <= 2 and len(current_words) != 0:

                if len(current_words) != len(current_labels):
                    print("Error")

                #if len(current_words) >= 512:
                #    print("Length error! Sequence truncated")
                #    current_words = current_words[:512]
                #    current_labels = current_labels[:512]

                total_words.append(current_words)
                total_labels.append(current_labels)

                current_words = []
                current_labels = []

            elif len(line_split) > 2:

                current_words.append(line_split[0])
                current_labels.append(line_split[3].replace("\n", ""))

    return total_words, total_labels

############################################################

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    if len(labels) != len(examples["tokens"]):
        print("Labels length unequal to tokenized inputs length")

    tokenized_inputs["labels"] = labels

    #print("tokenized_inputs keys")
    #print(tokenized_inputs.keys())

    ################################################

    if len(tokenized_inputs['input_ids'][0]) > 256:
    	tokenized_inputs["labels"] = [sub_label_list[:256] for sub_label_list in labels]
    	tokenized_inputs["input_ids"] = [sub_label_list[:256] for sub_label_list in tokenized_inputs["input_ids"]]
    	tokenized_inputs["attention_mask"] = [sub_label_list[:256] for sub_label_list in tokenized_inputs["attention_mask"]]
    else:
    	
    	new_labels = []
    	for sub_label_list in labels:
    		new_label_sub_list = sub_label_list
    		while len(new_label_sub_list) < 256:
    			new_label_sub_list.append(-100)
    		new_labels.append(new_label_sub_list)

    	new_input_ids = []
    	for sub_list in tokenized_inputs["input_ids"]:
    		new_sub_list = sub_list
    		while len(new_sub_list) < 256:
    			new_sub_list.append(0)
    		new_input_ids.append(new_sub_list)

    	new_attention_ids = []
    	for sub_list in tokenized_inputs["attention_mask"]:
    		new_sub_list = sub_list
    		while len(new_sub_list) < 256:
    			new_sub_list.append(0)
    		new_attention_ids.append(new_sub_list)

    	tokenized_inputs["labels"] = new_labels
    	tokenized_inputs["input_ids"] = new_input_ids
    	tokenized_inputs["attention_mask"] = new_attention_ids

    ################################################

    return tokenized_inputs

############################################################


device = "cuda:0"
device = torch.device(device)

num_epochs = 100 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
average_hidden_state = False
validation_set_scoring = False

assigned_batch_size = 8
gradient_accumulation_multiplier = 4

############################################################

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']
dataset = "bc5cdr"

model_choice = "roberta-large"

chosen_checkpoint_path = "paper_results_ner/roberta-large/bc5cdr/0.0001_18_True_10False_Run_0.pt"

half_configuration = True

############################################################

tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512, add_prefix_space=True)

############################################################

# Gather train, dev, and test sets
train_set_text, train_set_label = process_NER_dataset('ner/' + dataset + '/train.txt')

dev_set_text, dev_set_label = process_NER_dataset('ner/' + dataset + '/dev.txt')

test_set_text, test_set_label = process_NER_dataset('ner/' + dataset + '/test.txt')


####################################################################################

consolidated_labels = [label for label_list in train_set_label for label in label_list]

labels_list = sorted(list(set(consolidated_labels)))

print("Before reordering label list")
print(labels_list)

labels_list.insert(0, labels_list.pop(labels_list.index('O')))

print("After reordering label list")
print(labels_list)

label_to_value_dict = {}

count = 0
for label in labels_list:
  label_to_value_dict[label] = count
  count += 1

number_of_labels = len(list(label_to_value_dict.keys()))

print("Number of labels: " + str(number_of_labels))

####################################################################################

def convert_Label_to_Label_ID(label_list):

  new_list = []
  for label in label_list:
      new_list.append(label_to_value_dict[label])
  return new_list

####################################################################################

train_set_label = [convert_Label_to_Label_ID(label_list) for label_list in train_set_label]
dev_set_label = [convert_Label_to_Label_ID(label_list) for label_list in dev_set_label]
test_set_label = [convert_Label_to_Label_ID(label_list) for label_list in test_set_label]

print("Size of train, dev, and test")
print(len(train_set_label))
print(len(dev_set_label))
print(len(test_set_label))

####################################################################################

training_dataset_pandas = pd.DataFrame({'ner_tags': train_set_label, 'tokens': train_set_text})#[:1000]
training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

validation_dataset_pandas = pd.DataFrame({'ner_tags': dev_set_label, 'tokens': dev_set_text})#[:1000]
validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

test_dataset_pandas = pd.DataFrame({'ner_tags': test_set_label, 'tokens': test_set_text})
test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


############################################################

classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                'validation': validation_dataset_arrow, 
                                'test' : test_dataset_arrow})

tokenized_datasets = classification_dataset.map(tokenize_and_align_labels, batched=True, batch_size=assigned_batch_size)


#tokenized_datasets = tokenized_datasets.remove_columns(["tokens"])
tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


############################################################

lowest_recorded_validation_loss = 10000

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)


############################################################

model = AutoModelForTokenClassification.from_pretrained(model_choice, num_labels=number_of_labels, output_hidden_states=True)
#model = RobertaForSequenceClassification.from_pretrained(model_choice, num_labels=len(set(train_set_label)))

print("Loading the Best Model")

model.load_state_dict(torch.load(chosen_checkpoint_path))

model.to(device)

if half_configuration == True:
	print("Using Half Precision!")
	model.half()

############################################################

print("Beginning Evaluation")

metric = load_metric("accuracy")

total_predictions = torch.FloatTensor([]).to(device)
total_references = torch.FloatTensor([]).to(device)

progress_bar = tqdm(range(len(eval_dataloader)))
for batch in eval_dataloader:

    with torch.no_grad():

        new_batch = {'input_ids': batch['input_ids'].to(device),
                     'attention_mask': batch['attention_mask'].to(device)}
        labels = batch['labels'].to(device)

        outputs = model(**new_batch, labels=labels)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        total_predictions = torch.cat((total_predictions, torch.flatten(predictions)), 0)
        total_references = torch.cat((total_references, torch.flatten(labels)), 0)

        progress_bar.update(1)

######################################################################

new_total_predictions = []
new_total_references = []

for j in tqdm(range(0, len(total_predictions))):
    if total_references[j] != -100:
        new_total_predictions.append(total_predictions[j])
        new_total_references.append(total_references[j])

new_total_predictions = torch.FloatTensor(new_total_predictions)
new_total_references = torch.FloatTensor(new_total_references)

######################################################################

print("-----------------------------------------------------------------")

f_1_metric = load_metric("f1")
macro_f_1_results = f_1_metric.compute(average='macro', references=new_total_predictions, predictions=new_total_references)
print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
micro_f_1_results = f_1_metric.compute(average='micro', references=new_total_predictions, predictions=new_total_references)
print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))

############################################################