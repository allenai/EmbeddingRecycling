

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForSequenceClassification
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer, AutoModelForTokenClassification
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
          if model_choice == "roberta-large":

            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large" or model_choice == "microsoft/deberta-v3-xsmall":

            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 384
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice, output_hidden_states=True)
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

            elif model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':

                print("Number of Layers: " + str(len(list(self.encoderModel.roberta.encoder.layer))))

                layers_to_freeze = self.encoderModel.roberta.encoder.layer[:frozen_layer_count]
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
            if model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':
                for param in self.encoderModel.roberta.embeddings.parameters():
                    param.requires_grad = False

            else:
                for param in self.encoderModel.embeddings.parameters():
                    param.requires_grad = False


          


          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state

          #print("automodel structure")
          #other_model = AutoModelForTokenClassification.from_pretrained(model_choice, num_labels=number_of_labels, output_hidden_states=True)
          #print(other_model.__dict__)

          ############################################################################

          if delta_model_choice == 'BitFit':
                self.delta_model = BitFitModel(self.encoderModel)
                self.delta_model.freeze_module(exclude=unfrozen_components, set_state_dict=True)
                self.delta_model.log()
          elif delta_model_choice == 'Adapter':
                self.delta_model = AdapterModel(backbone_model=self.encoderModel, bottleneck_dim=bottleneck_value)
                self.delta_model.freeze_module(exclude=unfrozen_components, set_state_dict=True)
                self.delta_model.log()

          self.dropout = nn.Dropout(p=0.1, inplace=False)
          self.classifier = nn.Linear(in_features=embedding_size, out_features=number_of_labels, bias=True)

          #self.encoderModel = self.encoderModel.roberta

          
          #print("encoderModel")
          #print(self.encoderModel.__dict__)

          self.encoderModel.classifier = None



          

    def forward(self, ids, mask, labels):

            if model_choice == 'roberta-large':

                embeddings = self.encoderModel.roberta.embeddings(ids)
                extended_attention_mask = self.encoderModel.get_extended_attention_mask(mask, embeddings.size()[:-1], device)
                last_hidden_state = self.encoderModel.roberta.encoder(embeddings, extended_attention_mask)['last_hidden_state']

            else:

                embeddings = self.encoderModel.bert.embeddings(ids)
                extended_attention_mask = self.encoderModel.get_extended_attention_mask(mask, embeddings.size()[:-1], device)
                last_hidden_state = self.encoderModel.bert.encoder(embeddings, extended_attention_mask)['last_hidden_state']

            dropout_output = self.dropout(last_hidden_state)
            logits = self.classifier(dropout_output)

            ############################################

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = logits.view(-1, number_of_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, number_of_labels), labels.view(-1))

            ############################################

            return {'logits': logits, 'loss': loss} #total_output #total_output #classifier_output



############################################################

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

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']

num_epochs = 1000 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
number_of_runs = 10 #1 #5
frozen_choice = False
#chosen_learning_rate =  0.0001 #0.001, 0.0001, 1e-5, 5e-5, 5e-6
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False
validation_set_scoring = False

number_of_warmup_steps = 100

########################################################################################










############################################################
# Select model and hyperparameters here
############################################################

chosen_learning_rate_choices = [5e-5, 2e-5, 2e-4] # Learning rate choices for the bc5cdr, JNLPBA, 
                                                  # and NCBI-disease respectively
chosen_bottleneck_values = [256, 64, 256] # Bottleneck dimension choices for the Chemprot, SciCite, 
                                          # and SciERC-Relation respectively

delta_model_choice = 'Adapter' #'Adapter' #'BitFit'
 
model_choice = 'roberta-large'
#model_choice = 'allenai/scibert_scivocab_uncased'

############################################################











############################################################

assigned_batch_size = 32
tokenizer = AutoTokenizer.from_pretrained(model_choice, add_prefix_space=True)

############################################################

if model_choice == 'roberta-large':

	unfrozen_components = ['classifier']

	for i in range(12, 24):
		attention_adapter = 'encoder.layer.' + str(i) + ".attention.adapter"
		output_adapter = 'encoder.layer.' + str(i) + ".output.adapter"
		unfrozen_components.append(attention_adapter)
		unfrozen_components.append(output_adapter)

elif model_choice == 'allenai/scibert_scivocab_uncased':

	unfrozen_components = ['classifier']

	for i in range(6, 12):
		attention_adapter = 'encoder.layer.' + str(i) + ".attention.adapter"
		output_adapter = 'encoder.layer.' + str(i) + ".output.adapter"
		unfrozen_components.append(attention_adapter)
		unfrozen_components.append(output_adapter)


############################################################

dataset_folder_path = "paper_results_ner/"

if not os.path.isdir(dataset_folder_path):

	print("Creating folder: " + dataset_folder_path)
	os.mkdir(dataset_folder_path)

dataset_folder_path += model_choice.replace("/", "-") + "/"

if not os.path.isdir(dataset_folder_path):

    print("Creating folder: " + dataset_folder_path)
    os.mkdir(dataset_folder_path)

for dataset in classification_datasets:
    try:
        print("Making: " + dataset_folder_path + dataset)
        os.mkdir(dataset_folder_path + dataset)
    except:
        print("Already exists")
        print(dataset_folder_path + dataset)

############################################################

learning_rate_to_results_dict = {}

for chosen_learning_rate, bottleneck_value, dataset in zip(chosen_learning_rate_choices, chosen_bottleneck_values, classification_datasets):

        print("--------------------------------------------------------------------------")
        print("Starting new learning rate and bottleneck value: " + str(chosen_learning_rate) + " " + str(bottleneck_value))
        print("--------------------------------------------------------------------------")

        print("GPU Memory available at the start")
        print(get_gpu_memory())

        execution_start = time.time()

        print("Dataset: " + dataset)
        print("Model: " + model_choice)
        print("Dropout: " + str(current_dropout))
        print("Frozen Choice: " + str(frozen_choice))
        print("Number of Runs: " + str(number_of_runs))
        print('Learning Rate: ' + str(chosen_learning_rate))
        print("Number of Frozen Layers: " + str(frozen_layers))
        print("Frozen Embeddings: " + str(frozen_embeddings))
        print("Patience: " + str(patience_value))
        print("Average Hidden Layers: " + str(average_hidden_state))
        print("Validation Set Choice: " + str(validation_set_scoring))
        print("Number of Epochs: " + str(num_epochs))
        print("Bottleneck Value Choice: " + str(bottleneck_value))
        print("Batch Size: " + str(assigned_batch_size))
        print("Unfrozen Components: " + str(unfrozen_components))

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

            training_dataset_pandas = pd.DataFrame({'ner_tags': train_set_label, 'tokens': train_set_text})#[:1000]
            training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
            training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

            validation_dataset_pandas = pd.DataFrame({'ner_tags': dev_set_label, 'tokens': dev_set_text})#[:1000]
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


        ############################################################

        micro_averages = []
        macro_averages = []
        inference_times = []

        for i in range(0, number_of_runs):

            checkpoint_path = "paper_results_ner/" + model_choice.replace("/", "-") + "/" + dataset + "/experiment9_ner_" + str(chosen_learning_rate) + "_"
            checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
            checkpoint_path += str(validation_set_scoring) + "_Run_" + str(i) + ".pt"

            run_start = time.time()

            print("Loading Model")

            train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
            validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
            eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)


            ############################################################

            model = CustomBERTModel(number_of_labels, model_choice, current_dropout, 
                                    frozen_choice, frozen_layers, average_hidden_state, frozen_embeddings)
            #model = RobertaForSequenceClassification.from_pretrained(model_choice, num_labels=len(set(train_set_label)))
            
            model.to(device)

            ############################################################

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6
            #optimizer = Adam(model.parameters(), lr=1e-5) #5e-6

            num_training_steps = num_epochs * len(train_dataloader)

            lr_scheduler = get_scheduler(
                name="linear", optimizer=optimizer, num_warmup_steps=number_of_warmup_steps, num_training_steps=num_training_steps
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

                    new_batch = {'ids': batch['input_ids'].to(device),
                                 'mask': batch['attention_mask'].to(device),
                                 'labels': batch['labels'].to(device)}

                    outputs = model(**new_batch)

                    loss = outputs['loss']
                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    train_losses.append(loss.item())


                progress_bar = tqdm(range(len(validation_dataloader)))

                model.eval()
                for batch in validation_dataloader:

                        new_batch = {'ids': batch['input_ids'].to(device),
                                 	 'mask': batch['attention_mask'].to(device),
                                 	 'labels': batch['labels'].to(device)}

                        outputs = model(**new_batch)

                        loss = outputs['loss']
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
                early_stopping(valid_loss, model.delta_model)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break



            ############################################################

            print("Loading the Best Model")

            model.delta_model.load_state_dict(torch.load(checkpoint_path))



            ############################################################

            print("Beginning Evaluation")

            metric = load_metric("accuracy")

            total_predictions = torch.FloatTensor([]).to(device)
            total_references = torch.FloatTensor([]).to(device)

            inference_start = time.time()

            progress_bar = tqdm(range(len(eval_dataloader)))
            for batch in eval_dataloader:

                with torch.no_grad():

                    new_batch = {'ids': batch['input_ids'].to(device),
                                 'mask': batch['attention_mask'].to(device),
                                 'labels': batch['labels'].to(device)}

                    outputs = model(**new_batch)

                    logits = outputs['logits']

                    predictions = torch.argmax(logits, dim=-1)
                    labels = batch['labels'].to(device)

                    total_predictions = torch.cat((total_predictions, torch.flatten(predictions)), 0)
                    total_references = torch.cat((total_references, torch.flatten(labels)), 0)

                    progress_bar.update(1)



            inference_end = time.time()
            total_inference_time = inference_end - inference_start
            inference_times.append(total_inference_time)

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
            
            f_1_metric = load_metric("f1")
            macro_f_1_results = f_1_metric.compute(average='macro', references=new_total_references, predictions=new_total_predictions)
            print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
            micro_f_1_results = f_1_metric.compute(average='micro', references=new_total_references, predictions=new_total_predictions)
            print("Micro F1 for Test Set: " + str(micro_f_1_results['f1'] * 100))

            micro_averages.append(micro_f_1_results['f1'] * 100)
            macro_averages.append(macro_f_1_results['f1'] * 100)

        print("--------------------------------------------------")
        print("Final Results for Paper")
        print("--------------------------------------------------")
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
        print("--------------------------------------------------")

        ############################################################

