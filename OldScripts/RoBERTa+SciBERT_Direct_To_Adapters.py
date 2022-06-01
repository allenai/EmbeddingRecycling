

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForSequenceClassification
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
import tensorflow as tf
from opendelta import AdapterModel, BitFitModel

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
                 frozen_layer_count, average_hidden_state, frozen_embeddings, 
                 chosen_delta_model, chosen_frozen_components, chosen_bottleneck):

          super(CustomBERTModel, self).__init__()

          ####################################################################

          model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice, output_hidden_states=True)

          delta_model = AdapterModel(backbone_model=model_encoding, bottleneck_dim=chosen_bottleneck)
          delta_model.freeze_module(exclude=chosen_frozen_components, set_state_dict=True)
          delta_model.log()
          self.delta_model = delta_model

          embedding_size = 768
          self.encoderModel = model_encoding.bert.encoder

          ####################################################################


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
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 768)
                             )

          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, ids, mask, roberta_ids, roberta_mask):

          roberta_embeddings = finetuned_roberta_model(roberta_ids, attention_mask=roberta_mask)
          roberta_embeddings = roberta_embeddings['hidden_states'][0]
          roberta_embeddings_transformed = self.roberta_mlp(roberta_embeddings)

          scibert_embeddings = scibert_model_for_embeddings(ids, attention_mask=mask)
          scibert_embeddings = scibert_embeddings['hidden_states'][0]

          ################################################

          combined_embeddings = roberta_embeddings_transformed + scibert_embeddings
          
          total_output = self.encoderModel(combined_embeddings)
          scibert_output = total_output['last_hidden_state']
          scibert_output = scibert_output[:,0,:].view(-1, self.embedding_size)

          linear1_output = self.linear1(scibert_output)
          linear2_output = self.linear2(linear1_output)

          return linear2_output



############################################################

device = "cuda:0"
device = torch.device(device)

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction', 'mag']
#classification_datasets = ['sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']
#classification_datasets = ['mag']

num_epochs = 15 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
chosen_learning_rate = 1e-5 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = True

load_finetuned_roberta = False

random_state = 42


delta_model_choice = 'Adapter' #'Adapter' #'BitFit'
bottleneck_value = 24
unfrozen_components = ['deltas']


checkpoint_path = 'checkpoint_scibert_adapters_1331.pt' #'checkpoint38.pt' #'checkpoint36.pt' #'checkpoint34.pt'
model_choice = 'allenai/scibert_scivocab_uncased'
assigned_batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

scibert_model_for_embeddings = AutoModel.from_pretrained(model_choice, output_hidden_states=True).to(device)


############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

def roberta_tokenize_function(examples):

    tokenized_output = roberta_tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids
    tokenized_output['roberta_input_ids'] = tokenized_output.pop('input_ids')
    tokenized_output['roberta_attention_mask'] = tokenized_output.pop('attention_mask')

    return tokenized_output

############################################################

for dataset in classification_datasets:

    ###############################################################

    execution_start = time.time()

    print("GPU Memory available at the start")
    print(get_gpu_memory())

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
    print("Loading Finetuned Embeddings: " + str(load_finetuned_roberta))

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
    dev_label_list = sorted(list(set(dev_set_label)))
    test_label_list = sorted(list(set(test_set_label)))

    print("Label Lists")
    print(labels_list)
    print(dev_label_list)
    print(test_label_list)

    label_to_value_dict = {}

    count = 0
    for label in labels_list:
      label_to_value_dict[label] = count
      count += 1

    train_set_label = [label_to_value_dict[label] for label in train_set_label]
    dev_set_label = [label_to_value_dict[label] for label in dev_set_label]
    test_set_label = [label_to_value_dict[label] for label in test_set_label]

    print("Size of train, dev, and test sets")
    print(len(train_set_label))
    print(len(dev_set_label))
    print(len(test_set_label))

    ############################################################

    # Load pretrained, finetuned RoBERTa-Large encoder

    finetuned_roberta_model = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large', model_max_length=512)

    if load_finetuned_roberta == True:

        #finetuned_roberta_path = "../../../net/nfs2.s2-research/jons/prefinetuned_RoBERTa/new_pretrained_roberta-large_" + dataset + "_for_Scibert_mapping.pt"
        finetuned_roberta_path = "prefinetuned_RoBERTa/new_pretrained_roberta-large_" + dataset + "_for_Scibert_mapping.pt"
        finetuned_roberta_model.load_state_dict(torch.load(finetuned_roberta_path), strict=True)

    finetuned_roberta_model.to(device)

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
    
    tokenized_datasets = classification_dataset.map(roberta_tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

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
                                frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings,
                                delta_model_choice, unfrozen_components, bottleneck_value)

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

                #with torch.no_grad():
                
                    #batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch['labels'].to(device)

                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device), 
                                 'roberta_ids': batch['roberta_input_ids'].to(device),
                                 'roberta_mask': batch['roberta_attention_mask'].to(device)}
                    outputs = model(**new_batch)

                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    train_losses.append(loss.item())


            progress_bar = tqdm(range(len(validation_dataloader)))

            model.eval()
            for batch in validation_dataloader:

                #with torch.no_grad():
                
                    #batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch['labels'].to(device)

                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device), 
                                 'roberta_ids': batch['roberta_input_ids'].to(device),
                                 'roberta_mask': batch['roberta_attention_mask'].to(device)}
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

                #batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels'].to(device)

                new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device), 
                             'roberta_ids': batch['roberta_input_ids'].to(device),
                             'roberta_mask': batch['roberta_attention_mask'].to(device)}

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

        results = metric.compute(references=total_predictions, predictions=total_references)
        print("Accuracy for Test Set: " + str(results['accuracy']))

        f_1_metric = load_metric("f1")
        macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
        print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
        micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
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

