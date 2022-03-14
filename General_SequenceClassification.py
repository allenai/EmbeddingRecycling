
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



	model = BertForSequenceClassification.from_pretrained(model_choice, num_labels=len(set(train_set_label)))
	for param in model.bert.parameters():
		param.requires_grad = False

	device = torch.device("cuda:0")
	model.to(device)

	print("Number of labels: " + str(len(set(train_set_label))))

	############################################################


	optimizer = AdamW(model.parameters(), lr=0.001)

	num_epochs = 3
	num_training_steps = num_epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
	    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
	)

	############################################################

	print("Beginning Training")

	progress_bar = tqdm(range(num_training_steps))

	model.train()
	for epoch in range(num_epochs):
	    for batch in train_dataloader:
	        batch = {k: v.to(device) for k, v in batch.items()}
	        outputs = model(**batch)
	        loss = outputs.loss
	        loss.backward()

	        optimizer.step()
	        lr_scheduler.step()
	        optimizer.zero_grad()
	        progress_bar.update(1)


	############################################################

	print("Beginning Evaluation")

	metric = load_metric("accuracy")
	model.eval()

	total_predictions = torch.FloatTensor([]).to(device)
	total_references = torch.FloatTensor([]).to(device)

	for batch in eval_dataloader:
	    batch = {k: v.to(device) for k, v in batch.items()}
	    with torch.no_grad():
	        outputs = model(**batch)

	    logits = outputs.logits
	    predictions = torch.argmax(logits, dim=-1)
	    metric.add_batch(predictions=predictions, references=batch["labels"])

	    total_predictions = torch.cat((total_predictions, predictions), 0)
	    total_references = torch.cat((total_references, batch["labels"]), 0)



	print("--------------------------")
	print("Predictions Shapes")
	print(total_predictions.shape)
	print(total_references.shape)

	f_1_metric = load_metric("f1")
	macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
	print("Macro F1 for Test Set: " + str(macro_f_1_results['f1']))
	micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
	print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']))




