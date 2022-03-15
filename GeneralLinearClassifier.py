

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

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, embedding_size, dropout_layer, frozen):
          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice == "t5-3b":

            tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)
            model_encoding = T5EncoderModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          else:

            tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)
                                                      #attention_probs_dropout_prob=0.5)
                                                      #hidden_dropout_prob=0.5)
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding



          if frozen == True:
            for param in self.encoderModel.parameters():
                param.requires_grad = False




          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size


          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(
               ids, 
               attention_mask=mask)

          #pooler_output = total_output['pooler_output']
          sequence_output = total_output['last_hidden_state']

          #print('sequence_output[:,0,:].view(-1, self.embedding_size)')
          #print(sequence_output[:,0,:].view(-1, self.embedding_size).shape)

          linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size))


          #dropout = nn.Dropout(p=0.1)
          #linear1_output = dropout(linear1_output)

          #linear1_output = self.linear1(pooler_output) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)

          return linear2_output



############################################################

device = "cuda:0"

#classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
classification_datasets = ['sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']

num_epochs = 100 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
number_of_runs = 5 #1 #5
frozen_choice = False
chosen_learning_rate = 5e-5 #5e-6, 1e-5, 2e-5, 5e-5


#checkpoint_path = 'checkpoint1.pt'
#model_choice = "t5-3b"
#assigned_batch_size = 2

#checkpoint_path = 'checkpoint2.pt'
#model_choice = 'bert-base-uncased'
#assigned_batch_size = 32

checkpoint_path = 'checkpoint3.pt'
model_choice = 'allenai/scibert_scivocab_uncased'
assigned_batch_size = 32



#model_choice = 'hivemind/gpt-j-6B-8bit'
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 4096



############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################


for dataset in classification_datasets:

    print("Processing " + dataset + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout with " + str(number_of_runs) + " runs.")

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

    training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

    validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:1000]
    validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
    validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


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

    for i in range(0, number_of_runs):

        print("Loading Model")

        train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=assigned_batch_size)
        validation_dataloader = DataLoader(tokenized_datasets['validation'], shuffle=True, batch_size=assigned_batch_size)
        eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

        print("Number of labels: " + str(len(set(train_set_label))))

        ############################################################

        model = CustomBERTModel(len(set(train_set_label)), model_choice, embedding_size, current_dropout, frozen_choice)

        device = torch.device("cuda:0")
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
            for batch in train_dataloader:

                #with torch.no_grad():
                
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch['labels']

                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
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



        ############################################################

        print("Beginning Evaluation")

        metric = load_metric("accuracy")
        #model.eval()

        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)

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
    if len(micro_averages) > 0:
        print("Micro F1 Standard Variation: " + str(statistics.stdev(micro_averages)))

    print('macro_averages: ' + str(macro_averages))
    print("Macro F1 Average: " + str(statistics.mean(macro_averages)))
    if len(macro_averages) > 0:
        print("Macro F1 Standard Variation: " + str(statistics.stdev(macro_averages)))

