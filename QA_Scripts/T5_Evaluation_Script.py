

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

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


#import torch
#import torch_xla
#import torch_xla.core.xla_model as xm

import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm

###############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

#model_choice = "t5-base"
#output_directory = "models/tpu"
#path_for_args = "args.json"

model_choice = "google/t5-large-lm-adapt"
output_directory = "models/t5-large"
path_for_args = "t5_large_args.json"

print("Chosen Model: " + str(model_choice))

args_dict = {
  'training_script': 'train_t5_squad.py',
  "model_name_or_path": model_choice,
  "max_len": 512 ,
  "target_max_len": 16,
  "output_dir": output_directory,
  "overwrite_output_dir": True,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1e-4,
  "num_train_epochs": 4,
  "do_train": True,
  "remove_unused_columns": False,
  "n_gpu": 1
}

with open(path_for_args, 'w') as f:
  json.dump(args_dict, f)

tokenizer = T5Tokenizer.from_pretrained(model_choice)

###############################################################

# process the examples in input and target text format and the eos token at the end 
def add_eos_to_examples(example):
    example['input_text'] = 'question: %s  context: %s' % (example['question'], example['context'])
    example['target_text'] = '%s' % example['answers']['text'][0]
    return example

# tokenize the examples
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

# load train and validation split of squad
train_dataset  = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
non_torch_valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

#train_dataset  = nlp.load_dataset('squad', split="train[:100]")
#valid_dataset = nlp.load_dataset('squad', split="validation[:100]")
#non_torch_valid_dataset = nlp.load_dataset('squad', split="validation[:100]")

# map add_eos_to_examples function to the dataset example wise 
#train_dataset = train_dataset.map(add_eos_to_examples)
# map convert_to_features batch wise
#train_dataset = train_dataset.map(convert_to_features, batched=True)

valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

non_torch_valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
non_torch_valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


"""## Train"""

"""Let's write the arguments in a dict and store in a json file. The above code will load this file and parse the arguments."""

"""Start training!"""

#import torch_xla.distributed.xla_multiprocessing as xmp
#xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')


"""## Eval

There are two gotchas here. First the metrics functionality in the nlp package is still work-in-progress so we will use the official squad evaluation script. Second, for some reason which I couldn't figure out, the `.generate` method is not working on TPU so will need to do prediction on CPU. For predicting the validation set it almost takes 40 mins.
"""

## SQuAD evaluation script. Modifed slightly for this notebook


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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


    #return white_space_fix(remove_articles(remove_punc(lower(s))))
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

model = T5ForConditionalGeneration.from_pretrained(output_directory).to(device) # because its loaded on xla by default
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


final_results = evaluate(references, predictions)

print("----------------------------------------")
print("Final results")
print("----------------------------------------")
print(final_results['exact_match'])
print(final_results['f1'])
print("----------------------------------------")
