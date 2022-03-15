
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification

import pandas as pd
import numpy as np
import ast
import datasets
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import pyarrow as pa
import pyarrow.dataset as ds

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

import torch
from tqdm.auto import tqdm

device = "cuda:0"

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
model_choice = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)

checkpoint_path = 'checkpoint10.pt'
num_epochs = 100 #1000 #10
patience_value = 10 #10 #3

############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################


for dataset in classification_datasets:

    print("Processing " + dataset + " using " + model_choice + " with patience " + str(patience_value) + " on " + str(num_epochs) + " epochs.")

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


    dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                    'validation': validation_dataset_arrow, 
                                    'test' : test_dataset_arrow})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")


    ############################################################

    print("Loading Model")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=32)
    validation_dataloader = DataLoader(tokenized_datasets['validation'], shuffle=True, batch_size=32)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=32)



    model = AutoModelForSequenceClassification.from_pretrained(model_choice, num_labels=len(set(train_set_label)))
    #for param in model.bert.parameters():
    #	param.requires_grad = False

    device = torch.device("cuda:0")
    model.to(device)

    print("Number of labels: " + str(len(set(train_set_label))))

    ############################################################


    #optimizer = AdamW(model.parameters(), lr=5e-5)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    #lr_scheduler = get_scheduler(
    #    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    #)

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

                outputs = model(**batch)

                loss = outputs.loss

                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                train_losses.append(loss.item())


        progress_bar = tqdm(range(len(validation_dataloader)))

        model.eval()
        for batch in validation_dataloader:

            #with torch.no_grad():
            
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels']

                outputs = model(**batch)

                loss = outputs.loss
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

            outputs = model(**batch)

            logits = outputs.logits
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
    print("Results for Test Set: " + str(results['accuracy']))

    f_1_metric = load_metric("f1")
    macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
    print("Macro F1 for Test Set: " + str(macro_f_1_results['f1']))
    micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
    print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']))







