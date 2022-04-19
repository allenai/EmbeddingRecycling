

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, RobertaForSequenceClassification
from transformers import BertModel, AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaForTokenClassification, AutoModelForTokenClassification
import tensorflow as tf
from opendelta import AdapterModel, BitFitModel

import pandas as pd
import numpy as np
import ast
import datasets
from datasets import load_metric
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

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

                if len(current_words) >= 512:
                    print("Length error! Sequence truncated")
                    current_words = current_words[:512]
                    current_labels = current_labels[:512]

                total_words.append(current_words)
                total_labels.append(current_labels)

                current_words = []
                current_labels = []

            elif len(line_split) > 2:

                current_words.append(line_split[0])
                current_labels.append(line_split[3].replace("\n", ""))

    return total_words, total_labels



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

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

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

############################################################

device = "cuda:0"
device = torch.device(device)

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']
#classification_datasets = ['NCBI-disease']

num_epochs = 15 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
chosen_learning_rate =  2e-5 #0.001, 0.0001, 1e-5, 5e-5, 5e-6
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False
validation_set_scoring = False
 
checkpoint_path = 'checkpoint703.pt' # 41, 42, 43, 44, 45, 46, 47, 48, 49
model_choice = 'roberta-large'
assigned_batch_size = 2
tokenizer = AutoTokenizer.from_pretrained(model_choice, add_prefix_space=True)

#checkpoint_path = 'checkpoint_scibert_ner_2102.pt' # 41, 42, 43, 44, 45, 46, 47, 48, 49
#model_choice = 'allenai/scibert_scivocab_uncased'
#assigned_batch_size = 2 #16
#tokenizer = AutoTokenizer.from_pretrained(model_choice, add_prefix_space=True, model_max_length=512)

############################################################

for dataset in classification_datasets:

    print("GPU Memory available at the start")
    print(get_gpu_memory())

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

    # Gather train, dev, and test sets
    train_set_text, train_set_label = process_NER_dataset('ner/' + dataset + '/train.txt')

    dev_set_text, dev_set_label = process_NER_dataset('ner/' + dataset + '/dev.txt')

    test_set_text, test_set_label = process_NER_dataset('ner/' + dataset + '/test.txt')


    ####################################################################################

    consolidated_labels = [label for label_list in train_set_label for label in label_list]

    labels_list = sorted(list(set(consolidated_labels)))

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

    if validation_set_scoring == True:

        training_df = pd.DataFrame({'ner_tags': train_set_label, 'tokens': train_set_text})
        train, validation = train_test_split(training_df, test_size=0.15, shuffle=True)
        train.reset_index(drop=True, inplace=True)
        validation.reset_index(drop=True, inplace=True)

        training_dataset_pandas = train#[:1000]
        training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
        training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

        validation_dataset_pandas = validation#[:1000]
        validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
        validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

        test_dataset_pandas = pd.DataFrame({'ner_tags': dev_set_label, 'tokens': dev_set_text})
        test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
        test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    else:

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


    tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)

    tokenized_classification_dataset = classification_dataset.map(tokenize_and_align_labels, batched=True)


    from transformers import DataCollatorForTokenClassification

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model = AutoModelForTokenClassification.from_pretrained('roberta-large', num_labels=number_of_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=5e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_classification_dataset["train"],
        eval_dataset=tokenized_classification_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stop_callback]
    )

    #trainer.train()

    print("--------------------------------------------------")

    current_predictions = trainer.predict(tokenized_classification_dataset["test"])

    current_predicted_labels = current_predictions[1]
    current_reference_labels = tokenized_classification_dataset["test"]['labels']

    total_predictions = current_predicted_labels#.reshape(current_predicted_labels.shape[1])
    total_references = current_reference_labels#.reshape(current_reference_labels.shape[1])

    print("--------------------------")
    print("Predictions Shapes")
    print(total_predictions.shape)
    print(len(total_references))
    print(len(total_references[0]))

    f_1_metric = load_metric("f1")
    macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
    print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
    micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
    print("Micro F1 for Test Set: " + str(micro_f_1_results['f1'] * 100))

