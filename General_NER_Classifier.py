

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

    return tokenized_inputs

############################################################


device = "cuda:0"
device = torch.device(device)

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']
#classification_datasets = ['NCBI-disease']
#classification_datasets = ['JNLPBA', 'NCBI-disease']

num_epochs = 50 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 3 #1 #5
frozen_choice = False
#chosen_learning_rate =  0.0001 #0.001, 0.0001, 1e-5, 5e-5, 5e-6
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False
validation_set_scoring = True

random_state = 42

learning_rate_choices = [0.0001, 0.00001, 5e-5, 5e-6]
 
#checkpoint_path = 'checkpoint_roberta_ner_704.pt' # 41, 42, 43, 44, 45, 46, 47, 48, 49
#model_choice = 'roberta-large'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, add_prefix_space=True)

#checkpoint_path = 'checkpoint_scibert_ner_2111.pt' # 41, 42, 43, 44, 45, 46, 47, 48, 49
#model_choice = 'allenai/scibert_scivocab_uncased'
#assigned_batch_size = 32 #16
#tokenizer = AutoTokenizer.from_pretrained(model_choice, add_prefix_space=True, model_max_length=512)

#checkpoint_path = 'checkpoint_minilm_768_ner_115.pt'
#model_choice = 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512, add_prefix_space=True)

#checkpoint_path = 'checkpoint_minilm_384_ner_107.pt'
#model_choice = 'nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512, add_prefix_space=True)

checkpoint_path = 'checkpoint_distilbert_ner_114.pt'
model_choice = "distilbert-base-uncased"
assigned_batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#checkpoint_path = 'checkpoint_deberta_small_ner_37.pt' #'checkpoint38.pt' #'checkpoint36.pt' #'checkpoint34.pt'
#model_choice = 'microsoft/deberta-v3-small'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

#checkpoint_path = 'checkpoint_deberta_xsmall_ner_37.pt' #'checkpoint38.pt' #'checkpoint36.pt' #'checkpoint34.pt'
#model_choice = 'microsoft/deberta-v3-xsmall'
#assigned_batch_size = 32
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

############################################################

learning_rate_to_results_dict = {}

for chosen_learning_rate in learning_rate_choices:

    print("--------------------------------------------------------------------------")
    print("Starting new learning rate: " + str(chosen_learning_rate))
    print("--------------------------------------------------------------------------")

    current_learning_rate_results = {}

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

        if validation_set_scoring == True:

            training_df = pd.DataFrame({'ner_tags': train_set_label, 'tokens': train_set_text})
            train, validation = train_test_split(training_df, test_size=0.15, shuffle=True, random_state=random_state)
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

        tokenized_datasets = classification_dataset.map(tokenize_and_align_labels, batched=True, batch_size=assigned_batch_size)


        #tokenized_datasets = tokenized_datasets.remove_columns(["tokens"])
        tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])
        #tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")


        #print("tokenized_datasets")
        #print(tokenized_datasets['train'].features)
        #print(type(tokenized_datasets['train']['input_ids']))
        #print(len(tokenized_datasets['train']['input_ids']))
        #print(len(tokenized_datasets['train']['labels']))

        #print("----------------------------------")
        #for index in range(0, 3):
            #print(tokenized_datasets['train']['tokens'][index])
            #print(tokenized_datasets['train']['input_ids'][index])
            #print(tokenized_datasets['train']['labels'][index])
            #print(tokenized_datasets['train']['attention_mask'][index])
            #print(len(tokenized_datasets['train']['input_ids'][index]))
            #print(len(tokenized_datasets['train']['labels'][index]))
            #print(tokenized_datasets['train']['attention_mask'][index])
            #print("----------------------------------")


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

            model = AutoModelForTokenClassification.from_pretrained(model_choice, num_labels=number_of_labels)
            #model = RobertaForSequenceClassification.from_pretrained(model_choice, num_labels=len(set(train_set_label)))

            if frozen_layers > 0:

                if model_choice == "distilbert-base-uncased":

                    #print(model.__dict__)
                    print("Number of Layers: " + str(len(list(model.distilbert.transformer.layer))))
                    print("Number of Layers to Freeze: " + str(frozen_layers))

                    layers_to_freeze = model.distilbert.transformer.layer[:frozen_layers]
                    for module in layers_to_freeze:
                        for param in module.parameters():
                            param.requires_grad = False

                elif model_choice == 'allenai/scibert_scivocab_uncased':

                    #print(model.__dict__)
                    print("Number of Layers: " + str(len(list(model.bert.encoder.layer))))
                    print("Number of Layers to Freeze: " + str(frozen_layers))

                    layers_to_freeze = model.bert.encoder.layer[:frozen_layers]
                    for module in layers_to_freeze:
                        for param in module.parameters():
                            param.requires_grad = False

                else:

                    #print(model.__dict__)
                    print("Number of Layers: " + str(len(list(model.roberta.encoder.layer))))
                    print("Number of Layers to Freeze: " + str(frozen_layers))

                    layers_to_freeze = model.roberta.encoder.layer[:frozen_layers]
                    for module in layers_to_freeze:
                        for param in module.parameters():
                            param.requires_grad = False

            if frozen_embeddings == True:
                print("Frozen Embeddings Layer")
                if model_choice == "distilbert-base-uncased":
                    for param in model.distilbert.embeddings.parameters():
                        param.requires_grad = False
                elif model_choice == 'allenai/scibert_scivocab_uncased':
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = False
                else:
                    for param in model.roberta.embeddings.parameters():
                        param.requires_grad = False


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


                model.train()
                for batch in train_dataloader:

                        new_batch = {'input_ids': batch['input_ids'].to(device),
                                     'attention_mask': batch['attention_mask'].to(device)}
                        labels = batch['labels'].to(device)

                        outputs = model(**new_batch, labels=labels)

                        loss = outputs.loss
                        loss.backward()

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        train_losses.append(loss.item())


                progress_bar = tqdm(range(len(validation_dataloader)))

                model.eval()
                for batch in validation_dataloader:

                        new_batch = {'input_ids': batch['input_ids'].to(device),
                                     'attention_mask': batch['attention_mask'].to(device)}
                        labels = batch['labels'].to(device)

                        outputs = model(**new_batch, labels=labels)

                        loss = outputs.loss
                        loss.backward()

                        progress_bar.update(1)
                        train_losses.append(loss.item())
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

            total_predictions = torch.FloatTensor([]).to(device)
            total_references = torch.FloatTensor([]).to(device)

            inference_start = time.time()

            progress_bar = tqdm(range(len(eval_dataloader)))
            for batch in eval_dataloader:

                with torch.no_grad():

                    new_batch = {'input_ids': batch['input_ids'].to(device),
                                 'attention_mask': batch['attention_mask'].to(device)}
                    labels = batch['labels'].to(device)

                    outputs = model(**new_batch, labels=labels)

                    logits = outputs.logits

                    predictions = torch.argmax(logits, dim=-1)

                    #print("actual labels")
                    #print(labels.shape)
                    #print(torch.flatten(labels).shape)
                    #print("logits shape")
                    #print(predictions.shape)
                    #print(torch.flatten(predictions).shape)

                    total_predictions = torch.cat((total_predictions, torch.flatten(predictions)), 0)
                    total_references = torch.cat((total_references, torch.flatten(labels)), 0)

                    progress_bar.update(1)



            inference_end = time.time()
            total_inference_time = inference_end - inference_start
            inference_times.append(total_inference_time)

            ############################################################

            #print("--------------------------")
            #print("Predictions Shapes")
            #print(type(total_predictions))
            #print(type(total_references))
            #print(total_predictions.shape)
            #print(total_references.shape)
            #print(total_predictions[:30])
            #print(total_references[:30])

            ############################################################

            # Remove all the -100 references and predictions for calculating macro f-1 scores

            new_total_predictions = []
            new_total_references = []

            for j in tqdm(range(0, len(total_predictions))):
                if total_references[j] != -100:
                    new_total_predictions.append(total_predictions[j])
                    new_total_references.append(total_references[j])

            new_total_predictions = torch.FloatTensor(new_total_predictions)
            new_total_references = torch.FloatTensor(new_total_references)

            ############################################################

            #print("--------------------------")
            #print("Predictions Shapes after filtering out -100s")
            #print(type(new_total_predictions))
            #print(type(new_total_references))
            #print(new_total_predictions.shape)
            #print(new_total_references.shape)
            #print(new_total_predictions[:30])
            #print(new_total_references[:30])

            f_1_metric = load_metric("f1")
            macro_f_1_results = f_1_metric.compute(average='macro', references=new_total_predictions, predictions=new_total_references)
            print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
            micro_f_1_results = f_1_metric.compute(average='micro', references=new_total_predictions, predictions=new_total_references)
            print("Micro F1 for Test Set: " + str(micro_f_1_results['f1'] * 100))

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
print("Final Results for Learning Rate Tuning")
print("-----------------------------------------------------------------")

lr_sum_dict = {}

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

    ("--------------------------------------------")
    ("--------------------------------------------")

max_key = max(lr_sum_dict, key=lr_sum_dict.get)

print("Max Key: " + str(max_key))

with open('general_ner_classifier_results/' + model_choice + '_' + checkpoint_path + '.json', 'w') as fp:
    json.dump(learning_rate_to_results_dict, fp)

