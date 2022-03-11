

import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer

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

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, encoder_model, embedding_size):
          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          self.encoderModel = encoder_model
          ### New layers:
          self.linear1 = nn.Linear(embedding_size, 256)
          self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          

    def forward(self, ids, mask):
          
          total_output = self.encoderModel(
               ids, 
               attention_mask=mask)


          sequence_output = total_output['last_hidden_state']
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size)) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)

          return linear2_output


############################################################

device = "cuda:0"

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
#classification_datasets = ['chemprot']
#classification_datasets = ['sci-cite']
#classification_datasets = ['sciie-relation-extraction']

model_choice = "t5-3b"
tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)
model_encoding = T5EncoderModel.from_pretrained(model_choice)
embedding_size = 1024

#model_choice = 'bert-base-uncased'
#tokenizer = AutoTokenizer.from_pretrained(model_choice)
#model_encoding = BertModel.from_pretrained(model_choice)
#embedding_size = 768

#model_choice = 'allenai/scibert_scivocab_uncased'
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=512)
#model_encoding = AutoModel.from_pretrained(model_choice)
#embedding_size = 768

#model_choice = 'hivemind/gpt-j-6B-8bit'
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#model_encoding = BertModel.from_pretrained(model_choice)
#embedding_size = 4096



############################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################


for dataset in classification_datasets:

    print("Processing " + dataset + " using " + model_choice)

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

    training_dataset_pandas = pd.DataFrame({'label': train_set_label + dev_set_label, 'text': train_set_text + dev_set_text})#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)


    dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 'test' : test_dataset_arrow})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")


    ############################################################

    print("Loading Model")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=6)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=6)

    print("Number of labels: " + str(len(set(train_set_label))))

    model = CustomBERTModel(len(set(train_set_label)), model_encoding, embedding_size)

    device = torch.device("cuda:0")
    model.to(device)

    ############################################################


    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    ############################################################

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ############################################################

    print("Beginning Training")

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            #with torch.no_grad():
            
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels']

                #print("Batch")
                #print(batch)
                
                new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
                outputs = model(**new_batch)

                #print("outputs")
                #print(type(outputs))
                #print((outputs.shape))
                #print(outputs[0])

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)


    ############################################################

    print("Beginning Evaluation")

    metric = load_metric("accuracy")
    model.eval()

    total_predictions = torch.FloatTensor([]).to(device)
    total_references = torch.FloatTensor([]).to(device)

    progress_bar = tqdm(range(num_training_steps))

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
    f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
    print("F1 for Test Set: " + str(f_1_results['f1']))

