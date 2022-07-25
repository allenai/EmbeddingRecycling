


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

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

num_epochs = 100 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 5 #1 #5
frozen_choice = False
#chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa, 48 for DeBERTa XXL
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = False
assigned_batch_size = 4
gradient_accumulation_multiplier = 8

num_warmup_steps = 100

classification_datasets = ["sciie-relation-extraction"]
chosen_learning_rates = [1e-4]


############################################################
 
model_choice = "google/t5-large-lm-adapt"
tokenizer = AutoTokenizer.from_pretrained(model_choice)

############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################

dataset_folder_path = "paper_results_text_classification/"
if not os.path.isdir(dataset_folder_path):

    print("Creating folder: " + dataset_folder_path)
    os.mkdir(dataset_folder_path)

dataset_folder_path = dataset_folder_path + model_choice.replace("/", "-")
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

for chosen_learning_rate, dataset in zip(chosen_learning_rates, classification_datasets):

        print("---------------------------------------")
        print("Starting on " + dataset)
        print("---------------------------------------")

        print("GPU Memory available at the start")
        print(get_gpu_memory())

        execution_start = time.time()

        print("Dataset: " + dataset)
        print("Model: " + model_choice)
        print("Dropout: " + str(current_dropout))
        print("Frozen Choice: " + str(frozen_choice))
        print("Number of Runs: " + str(number_of_runs))
        print('Learning Rate: ' + str(chosen_learning_rate))
        #print("Checkpoint Path: " + checkpoint_path)
        print("Number of Frozen Layers: " + str(frozen_layers))
        print("Frozen Embeddings: " + str(frozen_embeddings))
        print("Patience: " + str(patience_value))
        print("Average Hidden Layers: " + str(average_hidden_state))
        print("Validation Set Choice: " + str(validation_set_scoring))
        print("Number of Epochs: " + str(num_epochs))
        print("Number of warmup steps: " + str(num_warmup_steps))

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

        if validation_set_scoring == True:

            training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:100]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:100]
            validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
            validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

            test_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
            test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
            test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

        else:

            training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:100]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:100]
            validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
            validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

            test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})#[:100]
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

        micro_averages = []
        macro_averages = []
        inference_times = []

        for i in range(0, number_of_runs):

            checkpoint_path = "paper_results_text_classification/" + model_choice.replace("/", "-") + "/" + dataset + "/" + str(chosen_learning_rate) + "_"
            checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
            checkpoint_path += str(validation_set_scoring) + "_" + str(i) + ".pt"

            print("Checkpoint Path: " + checkpoint_path)

            run_start = time.time()

            print("Loading Model")

            train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
            validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
            eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

            print("Number of labels: " + str(len(set(train_set_label))))

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
                        			 'labels': batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)}

                        loss = model(input_ids=new_batch['input_ids'], attention_mask=new_batch['attention_mask'], labels=new_batch['labels']).loss

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
                        			 'labels': batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)}

                        loss = model(input_ids=new_batch['input_ids'], labels=new_batch['labels']).loss

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
                    labels = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)
                    #decoder_attention_mask = batch['decoder_attention_mask'].to(device)

                    outputs = model(input_ids=inputs,
                            		attention_mask=attention_mask,
                            		labels=labels)
                            		#decoder_attention_mask=b_target_mask)

                    #print("inference comparison")
                    #print(batch.keys())
                    #print(outputs.keys())
                    #print(torch.argmax(outputs.logits, dim=-1))
                    #print(torch.argmax(outputs.logits, dim=-1).shape)
                    #print(outputs.logits.shape)
                    #print(labels)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

                    total_predictions = torch.cat((total_predictions, predictions), 0)
                    total_references = torch.cat((total_references, batch['labels'].to(device)), 0)

                    progress_bar.update(1)



            inference_end = time.time()
            total_inference_time = inference_end - inference_start
            inference_times.append(total_inference_time)

            ############################################################

            total_predictions = total_predictions.reshape(len(total_predictions))

            print("--------------------------")
            print("Predictions Shapes")
            print(total_predictions.shape)
            print(total_references.shape)
            #print(total_predictions)
            #print(total_references)

            results = metric.compute(references=total_references, predictions=total_predictions)
            print("Accuracy for Test Set: " + str(results['accuracy']))

            f_1_metric = load_metric("f1")
            macro_f_1_results = f_1_metric.compute(average='macro', references=total_references, predictions=total_predictions)
            print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
            micro_f_1_results = f_1_metric.compute(average='micro', references=total_references, predictions=total_predictions)
            print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))

            micro_averages.append(micro_f_1_results['f1'] * 100)
            macro_averages.append(macro_f_1_results['f1'] * 100)


        print("Processing " + dataset + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout")
        print('micro_averages: ' + str(micro_averages))
        if len(micro_averages) > 1:
            print("Micro F1 Average: " + str(statistics.mean(micro_averages)))
            print("Micro F1 Standard Variation: " + str(statistics.stdev(micro_averages)))

        print('macro_averages: ' + str(macro_averages))
        if len(macro_averages) > 1:
            print("Macro F1 Average: " + str(statistics.mean(macro_averages)))
            print("Macro F1 Standard Variation: " + str(statistics.stdev(macro_averages)))

        print("Inference Time Average: " + str(statistics.mean(inference_times)))
        print("Dataset Execution Run Time: " + str((time.time() - execution_start) / number_of_runs))
        print("Epoch Average Time: " + str((time.time() - run_start) / total_epochs_performed))

        print("GPU Memory available at the end")
        print(get_gpu_memory())

        ############################################################

        print("----------------------------------------")
        print("Final Paper Results for " + dataset)
        print("----------------------------------------")
        print("Micro and Macro F1")
        print(round(statistics.mean(micro_averages), 2))
        print(round(statistics.mean(macro_averages), 2))
        print("----------------------------------------")
        print("Micro and Macro F1 StDs")
        print(round(statistics.stdev(micro_averages), 2))
        print(round(statistics.stdev(macro_averages), 2))
        print("----------------------------------------")






