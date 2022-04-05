
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

############################################################

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep, len(oldModuleList)):
        print(i)
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings, cutoff_layers):

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

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
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

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]

                print("Number of layers to freeze: " + str(len(layers_to_freeze)))

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
          self.number_of_hidden_layers = frozen_layer_count

          if cutoff_layers > 0:

            print("Original number of layers: " + str(len(self.encoderModel.encoder.layer)))
            print("Cutoff layers: " + str(cutoff_layers))
            #self.encoderModel.encoder.layer = self.encoderModel.encoder.layer[cutoff_layers:]

            self.encoderModel = deleteEncodingLayers(AutoModel.from_pretrained(model_choice), cutoff_layers)
            #self.encoderModel = self.encoderModel.encoder


          

    def forward(self, inputs_embeds):
          
          #total_output = self.encoderModel(inputs_embeds=inputs_embeds)
          total_output = self.encoderModel.encoder(inputs_embeds)

          #print("total_output")
          #print(total_output)

          sequence_output = total_output['last_hidden_state']

          if self.average_hidden_state == True:

            print("Averaging the hidden states")

            sequence_output = torch.mean(sequence_output, dim=1)
            linear1_output = self.linear1(sequence_output)

          else:

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

num_epochs = 15 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False

chosen_learning_rate = 0.001 # Already tried 0.0001, 5e-6 #0.001, 0.0001, 1e-5, 5e-5, 5e-6

frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False
validation_set_scoring = False
layer_cutoff_count = 12

 
#checkpoint_path = 'checkpoint17.pt' #11, 12, 13, 15, 17, 18
#model_choice = "t5-3b"
#assigned_batch_size = 2
#tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)

#checkpoint_path = 'checkpoint22.pt'
#model_choice = 'bert-base-uncased'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#checkpoint_path = 'checkpoint37.pt' #'checkpoint38.pt' #'checkpoint36.pt' #'checkpoint34.pt'
#model_choice = 'allenai/scibert_scivocab_uncased'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

checkpoint_path = 'checkpoint49.pt' # 42, 43, 44, 45, 46, 47, 48, 49
model_choice = 'roberta-large'
assigned_batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)
#first_phase_encoder = AutoModel.from_pretrained(model_choice, output_hidden_states=True)

#checkpoint_path = 'checkpoint105.pt' #'checkpoint44.pt'
#model_choice = 'sentence-transformers/sentence-t5-base'
#assigned_batch_size = 32
#tokenizer = SentenceTransformer(model_choice, device='cuda').tokenizer 

#checkpoint_path = 'checkpoint207.pt' #'checkpoint205.pt' #'checkpoint44.pt'
#model_choice = "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"
#assigned_batch_size = 4
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#model_choice = 'hivemind/gpt-j-6B-8bit'
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 4096



############################################################

def tokenize_function(examples):

    #current_tokenization = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    #current_embeddings = first_phase_encoder(**current_tokenization)

    #return current_tokenization, current_embeddings#.input_ids

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

# def gather_embeddings(examples):

#     current_tokens = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
#     current_embeddings = first_phase_encoder(**current_tokens)
    
#     #print(current_embeddings.shape)
#     #print(current_embeddings)
#     return current_embeddings


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



    preloaded_training_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_training.pt')

    preloaded_validation_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_validation.pt')

    preloaded_test_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_testing.pt')



    ############################################################



    micro_averages = []
    macro_averages = []
    inference_times = []

    for i in range(0, number_of_runs):

        print("Loading Model")

        train_dataloader = DataLoader(preloaded_training_tensors, batch_size=assigned_batch_size)
        train_dataloader_labels = DataLoader(torch.LongTensor(train_set_label), batch_size=assigned_batch_size)

        validation_dataloader = DataLoader(preloaded_validation_tensors, batch_size=assigned_batch_size)
        validation_dataloader_labels = DataLoader(torch.LongTensor(dev_set_label), batch_size=assigned_batch_size)

        eval_dataloader = DataLoader(preloaded_test_tensors, batch_size=assigned_batch_size)
        eval_dataloader_labels = DataLoader(torch.LongTensor(test_set_label), batch_size=assigned_batch_size)

        print("Number of labels: " + str(len(set(train_set_label))))

        ############################################################

        model = CustomBERTModel(len(set(train_set_label)), model_choice, current_dropout, 
                                frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings, layer_cutoff_count)

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

        for epoch in range(num_epochs):

            print("Current Epoch: " + str(epoch))

            progress_bar = tqdm(range(len(train_dataloader)))

            
            model.train()
            for batch, labels in zip(train_dataloader, train_dataloader_labels):

                #with torch.no_grad():

                    new_batch = {'inputs_embeds': batch.to(device)}
                    outputs = model(**new_batch)

                    loss = criterion(outputs, labels.to(device))

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    train_losses.append(loss.item())


            progress_bar = tqdm(range(len(validation_dataloader)))

            model.eval()
            for batch, labels in zip(validation_dataloader, validation_dataloader_labels):

                #with torch.no_grad():
                
                    new_batch = {'inputs_embeds': batch.to(device)}
                    outputs = model(**new_batch)

                    loss = criterion(outputs, labels.to(device))
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



        ############################################################

        print("Beginning Evaluation")

        metric = load_metric("accuracy")
        #model.eval()

        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)

        inference_start = time.time()

        #progress_bar = tqdm(range(len(eval_dataloader)))
        #for batch in eval_dataloader:

        set_for_testing = eval_dataloader
        labels_for_testing = eval_dataloader_labels

        if validation_set_scoring == True:
            print("Using validation set for scoring")
            set_for_testing = validation_dataloader
            labels_for_testing = validation_dataloader_labels

        progress_bar = tqdm(range(len(set_for_testing)))
        for batch, labels in zip(set_for_testing, labels_for_testing):

            with torch.no_grad():

                labels = labels.to(device)

                new_batch = {'inputs_embeds': batch.to(device)}
                outputs = model(**new_batch)

                loss = criterion(outputs, labels)

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

        results = metric.compute(references=total_predictions, predictions=total_references)
        print("Accuracy for Test Set: " + str(results['accuracy']))

        f_1_metric = load_metric("f1")
        macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
        print("Macro F1 for Test Set: " + str(macro_f_1_results['f1']))
        micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
        print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']))

        micro_averages.append(micro_f_1_results['f1'])
        macro_averages.append(macro_f_1_results['f1'])


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
    print("Dataset Execution Run Time: " + str(time.time() - execution_start))

    print("GPU Memory available at the end")
    print(get_gpu_memory())

