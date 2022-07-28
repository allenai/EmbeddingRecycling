

import json
from collections import Counter
import string
import re
import argparse
import json
import sys

import torch
import nlp
from transformers import T5Tokenizer

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction, AutoModelForSeq2SeqLM
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from opendelta import AdapterModel, BitFitModel

import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm

###############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

model_choice = "google/t5-large-lm-adapt"
output_directory = "/net/nfs.cirrascale/s2-research/jons/checkpoints/models/google/t5-large-lm-adapt/5e-06/12"

tokenizer = T5Tokenizer.from_pretrained(model_choice)

###############################################################


def add_eos_to_examples(example):
    example['input_text'] = 'question: %s  context: %s' % (example['question'], example['context'])
    example['target_text'] = '%s' % example['answers']['text'][0]
    return example
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings


valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
non_torch_valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

non_torch_valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
non_torch_valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


def normalize_answer(s):
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_padding_tokens(text):
    	return text.replace("<pad> ", "").replace(" <pad>", "").replace("</s>","").replace("  ", " ")


    return white_space_fix(remove_articles(remove_punc(lower(remove_padding_tokens(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

##############################################################

model = AutoModelForSeq2SeqLM.from_pretrained(output_directory)

model.to(device)

##############################################################

tokenizer = T5Tokenizer.from_pretrained(model_choice)

valid_dataset = torch.load('valid_data.pt')
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8)

answers = []
for batch in tqdm(dataloader):
  outs = model.generate(input_ids=batch['input_ids'].to(device), 
                        attention_mask=batch['attention_mask'].to(device),
                        max_length=16,
                        early_stopping=True)
  outs = [tokenizer.decode(ids) for ids in outs]
  answers.extend(outs)

predictions = []
references = []
for ref, pred in zip(non_torch_valid_dataset, answers):
  predictions.append(pred)
  references.append(ref['answers']['text'])

print("Predictions and references")
print(predictions[:10])
print(references[:10])
print([normalize_answer(prediction) for prediction in predictions[:10]])


##############################################################


final_results = evaluate(references, predictions)

print("----------------------------------------")
print("Final results")
print("----------------------------------------")
print(final_results['exact_match'])
print(final_results['f1'])
print("----------------------------------------")



