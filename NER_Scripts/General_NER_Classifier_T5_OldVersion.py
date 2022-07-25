


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

def process_NER_dataset(dataset_path):

    total_words = []
    total_labels = []
    total_word_types = []

    current_words = []
    current_labels = []
    current_word_types = []

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
                total_word_types.append(current_word_types)

                current_words = []
                current_labels = []
                current_word_types = []

            elif len(line_split) > 2:

                current_words.append(line_split[0])
                current_labels.append(line_split[3].replace("\n", ""))
                current_word_types.append(line_split[1])

    return total_words, total_labels, total_word_types

##################################################################

def create_T5_formatted_inputs(text, labels, word_type_labels):

    formatted_text_inputs = []
    final_decoder_inputs = []
    total_labels = []

    for text_input, given_labels, given_word_types in zip(text, labels, word_type_labels):

        if len(text_input) != len(given_labels) or len(text_input) != len(given_word_types):
            print("Unequal sizes of inputs")

        current_text = ""
        current_decoder_input = ""
        current_labels = ""

        for token, label, word_type in zip(text_input, given_labels, given_word_types):
			
            if word_type == 'PUNCT':
                current_text += token
                current_decoder_input += token
            elif label != 'O':

                if len(current_text) != 0:
                    current_text += " "
                current_text += token #+ " [" + label + "]"

                if len(current_decoder_input) != 0:
                    current_decoder_input += " "
                current_decoder_input += token + " *" + label + "*"

                current_labels = token + " " + label + "; "

            else:
                if len(current_text) != 0:
                    current_text += " "
                current_text += " " + token

                if len(current_decoder_input) != 0:
                    current_decoder_input += " "
                current_decoder_input += " " + token

        formatted_text_inputs.append(current_text)
        final_decoder_inputs.append(current_decoder_input)
        total_labels.append(current_labels)

    return formatted_text_inputs, final_decoder_inputs, total_labels

##################################################################

#def extractPredictions(given_outputs, given_labels):







##################################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease'] #["sciie-relation-extraction", "mag"]

num_epochs = 15 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
#chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa, 48 for DeBERTa XXL
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = False
assigned_batch_size = 2
gradient_accumulation_multiplier = 16

num_warmup_steps = 100

#learning_rate_choices = [2e-5]
learning_rate_choices = [3e-4, 1e-5, 2e-5, 5e-5, 5e-6]#[1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 2e-5, 5e-5, 5e-6]
#learning_rate_choices = [3e-5, 4e-5, 5e-5, 6e-5]

############################################################
 
model_choice = "google/t5-large-lm-adapt"
tokenizer = AutoTokenizer.from_pretrained(model_choice)

############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################

def tokenize_function_decoder_inputs(examples):

    tokenized_labels = tokenizer(examples["decoder_input"], padding="max_length", truncation=True)
    tokenized_labels['decoder_inputs'] = tokenized_labels.pop('input_ids')

    return tokenized_labels

############################################################

def tokenize_function_labels(examples):

    tokenized_labels = tokenizer(examples["label"], padding="max_length", truncation=True)
    tokenized_labels['labels'] = tokenized_labels.pop('input_ids')

    return tokenized_labels

############################################################

dataset_folder_path = "checkpoints/" + model_choice.replace("/", "-")
if not os.path.isdir(dataset_folder_path):

    print("Creating folder: " + dataset_folder_path)
    os.mkdir(dataset_folder_path)

for dataset in classification_datasets:
    try:
        os.mkdir(dataset_folder_path + "/" + dataset)
    except:
        print("Already exists")
        print(dataset_folder_path + "/" + dataset)

############################################################

learning_rate_to_results_dict = {}

for chosen_learning_rate in learning_rate_choices:

    print("--------------------------------------------------------------------------")
    print("Starting new learning rate: " + str(chosen_learning_rate))
    print("--------------------------------------------------------------------------")

    current_learning_rate_results = {}

    for dataset in classification_datasets:

        checkpoint_path = "checkpoints/" + model_choice.replace("/", "-") + "/" + dataset + "/" + str(chosen_learning_rate) + "_"
        checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
        checkpoint_path += str(validation_set_scoring) + ".pt"

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
        print("Number of warmup steps: " + str(num_warmup_steps))

        ############################################################

        # Gather train, dev, and test sets
        train_set_text, train_set_label, train_word_type_labels = process_NER_dataset('ner/' + dataset + '/train.txt')

        dev_set_text, dev_set_label, dev_word_type_labels = process_NER_dataset('ner/' + dataset + '/dev.txt')

        test_set_text, test_set_label, test_word_type_labels = process_NER_dataset('ner/' + dataset + '/test.txt')

        ############################################################

        train_T5_formatted_text_inputs, train_decoder_inputs, train_T5_labels = create_T5_formatted_inputs(train_set_text, train_set_label, train_word_type_labels)

        dev_T5_formatted_text_inputs, dev_decoder_inputs, dev_T5_labels = create_T5_formatted_inputs(dev_set_text, dev_set_label, dev_word_type_labels)

        test_T5_formatted_text_inputs, test_decoder_inputs, test_T5_labels = create_T5_formatted_inputs(test_set_text, test_set_label, test_word_type_labels)

        ############################################################

        if validation_set_scoring == True:

            training_dataset_pandas = pd.DataFrame({'label': train_T5_labels, 'decoder_input': train_decoder_inputs, 'text': train_T5_formatted_text_inputs})#[:100]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = pd.DataFrame({'label': dev_T5_labels, 'decoder_input': dev_decoder_inputs, 'text': dev_T5_formatted_text_inputs})#[:100]
            validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
            validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

            test_dataset_pandas = pd.DataFrame({'label': dev_T5_labels, 'decoder_input': dev_decoder_inputs, 'text': dev_set_text})
            test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
            test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

        else:

            training_dataset_pandas = pd.DataFrame({'label': train_T5_labels, 'decoder_input': train_decoder_inputs, 'text': train_T5_formatted_text_inputs})#[:100]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = pd.DataFrame({'label': dev_T5_labels, 'decoder_input': dev_decoder_inputs, 'text': dev_T5_formatted_text_inputs})#[:100]
            validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
            validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

            test_dataset_pandas = pd.DataFrame({'label': test_T5_labels, 'decoder_input': test_decoder_inputs, 'text': test_T5_formatted_text_inputs})#[:100]
            test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
            test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


        ############################################################


        classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                        'validation': validation_dataset_arrow, 
                                        'test' : test_dataset_arrow})
        tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.map(tokenize_function_decoder_inputs, batched=True)
        tokenized_datasets = tokenized_datasets.map(tokenize_function_labels, batched=True)


        tokenized_datasets = tokenized_datasets.remove_columns(["text", "decoder_input", "label"])
        #tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")


        ############################################################

        micro_averages = []
        macro_averages = []
        inference_times = []

        for i in range(0, number_of_runs):

            run_start = time.time()

            print("Loading Model")

            train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
            validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
            eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

            ############################################################

            model = T5ForConditionalGeneration.from_pretrained(model_choice)

            if frozen_layers > 0:

                print("Freezing T5-3b")
                print("Number of Layers: " + str(len(model.encoder.block)))

                for parameter in model.encoder.embed_tokens.parameters():
                    parameter.requires_grad = False

                for i, m in enumerate(model.encoder.block):        
                    #Only un-freeze the last n transformer blocks
                    if i+1 > 24 - frozen_layers:
                        print(str(i) + " Layer")
                        for parameter in m.parameters():
                            parameter.requires_grad = False

            ############################################################

            model.to(device)

            ############################################################


            #optimizer = AdamW(model.parameters(), lr=5e-5)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6
            #optimizer = Adam(model.parameters(), lr=1e-5) #5e-6

            num_training_steps = num_epochs * len(train_dataloader)

            lr_scheduler = get_scheduler(
                name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

            ############################################################



            # to track the training loss as the model trains
            train_losses = []
            # to track the validation loss as the model trains
            valid_losses = []
            # to track the average training loss per epoch as the model trains
            avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            avg_valid_losses = []


            # import EarlyStopping
            from pytorchtools import EarlyStopping
            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=patience_value, verbose=True, path=checkpoint_path)
            #early_stopping = EarlyStopping(patience=10, verbose=True)

            print("Checkpoint Path: " + checkpoint_path)


            print("Beginning Training")

            total_epochs_performed = 0

            for epoch in range(num_epochs):

                total_epochs_performed += 1

                print("Current Epoch: " + str(epoch))

                progress_bar = tqdm(range(len(train_dataloader)))


                gradient_accumulation_count = 0

                model.train()
                for batch in train_dataloader:

                    #with torch.no_grad():
                    
                        new_batch = {'input_ids': batch['input_ids'].to(device), 
                        			 'attention_mask': batch['attention_mask'].to(device),
                        			 'decoder_inputs': batch['decoder_inputs'].to(device)}

                        loss = model(input_ids=new_batch['input_ids'], attention_mask=new_batch['attention_mask'], labels=new_batch['decoder_inputs']).loss

                        loss.backward()

                        gradient_accumulation_count += 1
                        if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                        
                        progress_bar.update(1)
                        train_losses.append(loss.item())


                progress_bar = tqdm(range(len(validation_dataloader)))

                model.eval()
                for batch in validation_dataloader:

                    #with torch.no_grad():
                    
                        new_batch = {'input_ids': batch['input_ids'].to(device), 
                        			 'attention_mask': batch['attention_mask'].to(device),
                        			 'decoder_inputs': batch['decoder_inputs'].to(device)}

                        loss = model(input_ids=new_batch['input_ids'], attention_mask=new_batch['attention_mask'], labels=new_batch['decoder_inputs']).loss

                        progress_bar.update(1)

                        valid_losses.append(loss.item())


                # print training/validation statistics 
                # calculate average loss over an epoch
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)
                
                epoch_len = len(str(num_epochs))
                
                print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.5f} ' +
                             f'valid_loss: {valid_loss:.5f}')
                
                print(print_msg)
                
                # clear lists to track next epoch
                train_losses = []
                valid_losses = []
                
                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, model)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break



            ############################################################

            print("Loading the Best Model")

            model.load_state_dict(torch.load(checkpoint_path))

            #torch.save(model.encoderModel.state_dict(), model_encoder_path)



            ############################################################

            print("Beginning Evaluation")

            metric = load_metric("accuracy")
            #model.eval()

            total_predictions = torch.FloatTensor([]).to(device)
            total_references = torch.FloatTensor([]).to(device)

            inference_start = time.time()

            #progress_bar = tqdm(range(len(eval_dataloader)))
            #for batch in eval_dataloader:

            progress_bar = tqdm(range(len(eval_dataloader)))
            for batch in eval_dataloader:

                with torch.no_grad():

                    inputs = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    decoder_inputs = batch['decoder_inputs'].to(device)
                    #decoder_attention_mask = batch['decoder_attention_mask'].to(device)

                    output = model.generate(input_ids=inputs, 
                                            attention_mask=attention_mask,
                                            max_length=512)

                    print("---------------------------------------------------")
                    print("Model output")
                    print(inputs.shape)
                    print(output.logits)
                    print(tokenizer.decode(inputs[0], skip_special_tokens=True))
                    print(tokenizer.decode(output[0], skip_special_tokens=True))
                    print(tokenizer.decode(labels[0], skip_special_tokens=True))
                    print("---------------------------------------------------")

                    predictions, references = extractPredictions(output, labels)




