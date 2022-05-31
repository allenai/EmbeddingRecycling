

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

device = "cuda:0"
device = torch.device(device)

classification_datasets = ['chemprot', 'sci-cite', "sciie-relation-extraction"] #["sciie-relation-extraction", "mag"]

num_epochs = 1 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
number_of_runs = 3 #1 #5
frozen_choice = False
#chosen_learning_rate = 0.0001 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = True
assigned_batch_size = 8
gradient_accumulation_multiplier = 4


#learning_rate_choices = [2e-5]
learning_rate_choices = [0.0001, 1e-5, 2e-5, 5e-5, 5e-6]
#learning_rate_choices = [3e-5, 4e-5, 5e-5, 6e-5]

############################################################
 
#model_choice = "t5-3b"
#tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'bert-base-uncased'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'allenai/scibert_scivocab_uncased'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'roberta-large'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'microsoft/deberta-v3-small'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'microsoft/deberta-v3-xsmall'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'distilroberta-base'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'sentence-transformers/sentence-t5-base'
#tokenizer = SentenceTransformer(model_choice, device='cuda').tokenizer 

#model_choice = 'nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

model_choice = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = "t5-small"
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 4096



############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

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

            training_df = pd.DataFrame({'label': train_set_label, 'text': train_set_text})
            train, validation = train_test_split(training_df, test_size=0.15, shuffle=True, random_state=random_state)
            train.reset_index(drop=True, inplace=True)
            validation.reset_index(drop=True, inplace=True)

            training_dataset_pandas = train#[:1000]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = validation#[:1000]
            validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
            validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

            test_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
            test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
            test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

        else:

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

        micro_averages = []
        macro_averages = []
        inference_times = []

        for i in range(0, number_of_runs):

            run_start = time.time()

            print("Loading Model")

            train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
            validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
            eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

            print("Number of labels: " + str(len(set(train_set_label))))

            ############################################################

            model = CustomBERTModel(len(set(train_set_label)), model_choice, current_dropout, 
                                    frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)

            model.to(device)

            ############################################################


            #optimizer = AdamW(model.parameters(), lr=5e-5)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6
            #optimizer = Adam(model.parameters(), lr=1e-5) #5e-6

            num_training_steps = num_epochs * len(train_dataloader)

            lr_scheduler = get_scheduler(
                name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
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
                    
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch['labels']

                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
                        outputs = model(**new_batch)

                        #print("Example outputs")
                        #print(outputs)
                        #print(labels)

                        loss = criterion(outputs, labels)

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
                    
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch['labels']

                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
                        outputs = model(**new_batch)

                        loss = criterion(outputs, labels)
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

                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch['labels']

                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                    outputs = model(**new_batch)

                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=labels)

                    total_predictions = torch.cat((total_predictions, predictions), 0)
                    total_references = torch.cat((total_references, labels), 0)

                    progress_bar.update(1)



            inference_end = time.time()
            total_inference_time = inference_end - inference_start
            inference_times.append(total_inference_time)

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

            micro_averages.append(micro_f_1_results['f1'] * 100)
            macro_averages.append(macro_f_1_results['f1'] * 100)


        print("Processing " + dataset + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout")
        print('micro_averages: ' + str(micro_averages))
        print("Micro F1 Average: " + str(statistics.mean(micro_averages)))
        if len(micro_averages) > 1:
            print("Micro F1 Standard Variation: " + str(statistics.stdev(micro_averages)))

        print('macro_averages: ' + str(macro_averages))
        print("Macro F1 Average: " + str(statistics.mean(macro_averages)))
        if len(macro_averages) > 1:
            print("Macro F1 Standard Variation: " + str(statistics.stdev(macro_averages)))

        print("Inference Time Average: " + str(statistics.mean(inference_times)))
        print("Dataset Execution Run Time: " + str((time.time() - execution_start) / number_of_runs))
        print("Epoch Average Time: " + str((time.time() - run_start) / total_epochs_performed))

        print("GPU Memory available at the end")
        print(get_gpu_memory())

        ############################################################

        current_learning_rate_results[dataset + "_micro_f1_average"] =  statistics.mean(micro_averages)
        current_learning_rate_results[dataset + "_micro_f1_std"] =  statistics.stdev(micro_averages)
        current_learning_rate_results[dataset + "_macro_f1_average"] =  statistics.mean(macro_averages)
        current_learning_rate_results[dataset + "_macro_f1_std"] =  statistics.stdev(macro_averages)

    ############################################################
    
    learning_rate_to_results_dict[str(chosen_learning_rate)] = current_learning_rate_results



















############################################################

print("-----------------------------------------------------------------")
print("Results for Learning Rate Tuning")
print("-----------------------------------------------------------------")

lr_sum_dict = {}

print(learning_rate_to_results_dict)

for chosen_learning_rate in learning_rate_choices:

    current_lr_sum = 0

    for dataset in classification_datasets:

        print("Results for " + str(chosen_learning_rate) + " for " + dataset)
        print(dataset + "_micro_f1_average: " + str(learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_micro_f1_average"]))
        print(dataset + "_micro_f1_std: " + str(learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_micro_f1_std"]))
        print(dataset + "_macro_f1_average: " + str(learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_macro_f1_average"]))
        print(dataset + "_macro_f1_std: " + str(learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_macro_f1_std"]))
        print("--------------------------------------------")

        current_lr_sum += learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_micro_f1_average"]
        current_lr_sum += learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_macro_f1_average"]

    lr_sum_dict[str(chosen_learning_rate)] = current_lr_sum

    print("--------------------------------------------")
    print("--------------------------------------------")


max_key = max(lr_sum_dict, key=lr_sum_dict.get)

print("Max Key: " + str(max_key))

saved_results_file_path = "Layers_Frozen_" + str(frozen_layers) + '_'
saved_results_file_path += "Runs_" + str(number_of_runs) + "_"
saved_results_file_path += "FrozenEmbeddings_" + str(frozen_embeddings) + '_'
saved_results_file_path += "ValidationScoring_" + str(validation_set_scoring) + '.json'

############################################################

if not os.path.isdir('general_linear_classifier_results'):
    os.mkdir('general_linear_classifier_results')

results_folder_path = "general_linear_classifier_results/" + model_choice.replace("/", "-")
if not os.path.isdir(results_folder_path):

    print("Creating folder: " + results_folder_path)
    os.mkdir(results_folder_path)

    for dataset in classification_datasets:
        os.mkdir(results_folder_path + "/" + dataset)

############################################################

with open('general_linear_classifier_results/' + model_choice.replace("/", "-") + "/" + saved_results_file_path, 'w') as fp:
    json.dump(learning_rate_to_results_dict, fp)















############################################################

print("-----------------------------------------------------------------")
print("Final Results: Best LR for each dataset")
print("-----------------------------------------------------------------")

dataset_to_best_lr_dict = {}

for dataset in classification_datasets:

    best_lr = learning_rate_choices[0]
    best_combined_f1 = [0, 0]
    best_combined_stds = [0, 0]

    for chosen_learning_rate in learning_rate_choices:

        current_combined_macro_micro_f1 = [learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_micro_f1_average"],
                                           learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_macro_f1_average"]]

        if sum(best_combined_f1) < sum(current_combined_macro_micro_f1):
            best_lr = chosen_learning_rate
            best_combined_f1 = current_combined_macro_micro_f1
            best_combined_stds = [learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_micro_f1_std"],
                                  learning_rate_to_results_dict[str(chosen_learning_rate)][dataset + "_macro_f1_std"]]

    dataset_to_best_lr_dict[dataset] = {
                                            'best_lr': best_lr,
                                            'best_combined_f1': best_combined_f1,
                                            'best_combined_stds': best_combined_stds
                                       }

    print("--------------------------------------------")
    print("Results for " + dataset)
    print("Best LR: " + str(dataset_to_best_lr_dict[dataset]['best_lr']))
    print("Best Micro F1: " + str(dataset_to_best_lr_dict[dataset]['best_combined_f1'][0]))
    print("Best Macro F1: " + str(dataset_to_best_lr_dict[dataset]['best_combined_f1'][1]))
    print("Micro StD: " + str(dataset_to_best_lr_dict[dataset]['best_combined_stds'][0]))
    print("Macro StD: " + str(dataset_to_best_lr_dict[dataset]['best_combined_stds'][1]))
    print("--------------------------------------------")

print("-----------------------------------------------------------------")
print("Final Results for Spreadsheet")
print("-----------------------------------------------------------------")
print("Dataset: " + dataset)
print("Model: " + model_choice)
print("Number of Runs: " + str(number_of_runs))
print("Number of Epochs: " + str(num_epochs))
print("Patience: " + str(patience_value))
print("Number of Frozen Layers: " + str(frozen_layers))
print("Frozen Embeddings: " + str(frozen_embeddings))
print("Validation Set Choice: " + str(validation_set_scoring))
print("-----------------------------------------------------------------")

print("Learning Rates")
for dataset in classification_datasets:

    print(str(dataset_to_best_lr_dict[dataset]['best_lr']))

print("-----------------------------------------------------------------")
print("Micro and Macro F1 Scores")
for dataset in classification_datasets:

    print(str(round(dataset_to_best_lr_dict[dataset]['best_combined_f1'][0], 2))) 
    print(str(round(dataset_to_best_lr_dict[dataset]['best_combined_f1'][1], 2))) 

print("-----------------------------------------------------------------")
print("Micro and Macro StDs")
for dataset in classification_datasets:

    print(str(round(dataset_to_best_lr_dict[dataset]['best_combined_stds'][0], 2))) 
    print(str(round(dataset_to_best_lr_dict[dataset]['best_combined_stds'][1], 2))) 




