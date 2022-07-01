
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, load_metric
from transformers import DefaultDataCollator, AutoTokenizer, get_scheduler, AutoModel
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm
import torch.nn as nn
from opendelta import AdapterModel, BitFitModel
import tensorflow as tf

from urllib.request import urlopen, Request

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import random
import torch
import os
import time
import statistics

from collections import Counter

##################################################

def compute_f1(predictions_list, references_list):

	f1_scores = []

	for index in range(0, len(predictions_list)):

	    prediction = predictions_list[index]
	    truth = references_list[index]

	    pred_tokens = [x for x in range(prediction[0], prediction[1] + 1)]
	    truth_tokens = [x for x in range(truth[0], truth[1] + 1)]

	    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
	    
	    # if there are no common tokens then f1 = 0
	    if len(common_tokens) == 0:
	        f1_scores.append(0)
	    else:
	    
		    prec = len(common_tokens) / len(pred_tokens)
		    rec = len(common_tokens) / len(truth_tokens)
	    
		    f1_scores.append(2 * (prec * rec) / (prec + rec))

	return statistics.mean(f1_scores)

##################################################

def exact_match(predictions_list, references_list):

	match_count = 0
	for prediction, truth in zip(predictions_list, references_list):
		if prediction[0] == truth[0] and prediction[1] == truth[1]:
			match_count += 1

	return match_count / len(predictions_list)

##################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

squad = load_dataset("squad")

from transformers import AutoTokenizer

#model_choice = "distilbert-base-uncased"
model_choice = 'csarron/bert-base-uncased-squad-v1'

tokenizer = AutoTokenizer.from_pretrained(model_choice)
model = AutoModelForQuestionAnswering.from_pretrained(model_choice)

model.to(device)

##################################################

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

##################################################

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
tokenized_squad.set_format("torch")

##################################################

assigned_batch_size = 8

train_dataloader = DataLoader(tokenized_squad['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(tokenized_squad['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(tokenized_squad['validation'], batch_size=assigned_batch_size)

##################################################

total_start_position_predictions = torch.LongTensor([]).to(device)
total_start_position_references = torch.LongTensor([]).to(device)

total_end_position_predictions = torch.LongTensor([]).to(device)
total_end_position_references = torch.LongTensor([]).to(device)

inference_start = time.time()

progress_bar = tqdm(range(len(eval_dataloader)))
for batch in eval_dataloader:

    with torch.no_grad():

        new_batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**new_batch)

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']

        start_predictions = torch.argmax(start_logits, dim=-1)
        end_predictions = torch.argmax(end_logits, dim=-1)

        total_start_position_predictions = torch.cat((total_start_position_predictions, start_predictions), 0)
        total_start_position_references = torch.cat((total_start_position_references, new_batch['start_positions']), 0)

        total_end_position_predictions = torch.cat((total_end_position_predictions, end_predictions), 0)
        total_end_position_references = torch.cat((total_end_position_references, new_batch['end_positions']), 0)

        progress_bar.update(1)



############################################################

total_start_position_predictions = total_start_position_predictions.tolist()
total_start_position_references = total_start_position_references.tolist()

total_end_position_predictions = total_end_position_predictions.tolist()
total_end_position_references = total_end_position_references.tolist()

############################################################

final_predictions = []
for index in range(0, len(total_start_position_predictions)):
	final_predictions.append([total_start_position_predictions[index], total_end_position_predictions[index]])

final_references = []
for index in range(0, len(total_start_position_references)):
	final_references.append([total_start_position_references[index], total_end_position_references[index]])

print("Final predictions lengths")
print(len(final_predictions))
print((final_predictions[100]))
print(len(final_references))
print((final_references[100]))

f1_score = compute_f1(final_predictions, final_references)

############################################################

exact_match_score = exact_match(final_predictions, final_references)

############################################################

print("Exact Match: " + str(round(exact_match_score * 100, 2)))
print("F1-Score: " + str(round(f1_score * 100, 2)))

