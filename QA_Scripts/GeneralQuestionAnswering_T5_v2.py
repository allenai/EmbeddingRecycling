

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


import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm
import statistics

###############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

#model_choice = "t5-base"
model_choice = "google/t5-large-lm-adapt"

frozen_layers = 0

runs = 3

learning_rate_choices = [5e-5]

exact_match_scores = []
f1_scores = []

###############################################################

results_string = ""

for learning_rate in learning_rate_choices:

	for run in range(0, runs):

		print("-------------------------------------------------------------")
		print("Chosen Model: " + str(model_choice))
		print("Learning Rate: " + str(learning_rate))
		print("Frozen Layer Count: " + str(frozen_layers))
		print("-------------------------------------------------------------")

		output_directory = "/net/nfs.cirrascale/s2-research/jons/checkpoints/models/" + model_choice + "/" + str(learning_rate) + "/" + str(frozen_layers)
		path_for_args = str(model_choice.replace("/", "-")) + "_" + str(frozen_layers) + "_" + str(learning_rate) + "_args.json"

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
		  "learning_rate": learning_rate,
		  "num_train_epochs": 4,
		  "do_train": True,
		  "remove_unused_columns": False,
		  "n_gpu": 1,
		  "cache_dir": output_directory,
		  "save_strategy": "no"
		}

		with open(path_for_args, 'w') as f:
		  json.dump(args_dict, f)

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

		train_dataset  = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
		valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
		non_torch_valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

		train_dataset = train_dataset.map(add_eos_to_examples)
		train_dataset = train_dataset.map(convert_to_features, batched=True)

		valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
		valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

		non_torch_valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
		non_torch_valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


		columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
		train_dataset.set_format(type='torch', columns=columns)
		valid_dataset.set_format(type='torch', columns=columns)

		torch.save(train_dataset, 'train_data.pt')
		torch.save(valid_dataset, 'valid_data.pt')

		print("Finishing Preprocessing")

		########################################################################################

		logger = logging.getLogger(__name__)

		@dataclass
		class T2TDataCollator():
		    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
		        

		        input_ids = torch.stack([example['input_ids'] for example in batch])
		        labels = torch.stack([example['target_ids'] for example in batch])
		        labels[labels[:, :] == 0] = -100
		        attention_mask = torch.stack([example['attention_mask'] for example in batch])
		        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
		        

		        return {
		            'input_ids': input_ids, 
		            'attention_mask': attention_mask,
		            'labels': labels, 
		            'decoder_attention_mask': decoder_attention_mask
		        }


		@dataclass
		class ModelArguments:
		    

		    model_name_or_path: str = field(
		        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
		    )
		    tokenizer_name: Optional[str] = field(
		        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
		    )
		    cache_dir: Optional[str] = field(
		        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
		    )

		@dataclass
		class DataTrainingArguments:
		    

		    train_file_path: Optional[str] = field(
		        default='train_data.pt',
		        metadata={"help": "Path for cached train dataset"},
		    )
		    valid_file_path: Optional[str] = field(
		        default='valid_data.pt',
		        metadata={"help": "Path for cached valid dataset"},
		    )
		    max_len: Optional[int] = field(
		        default=512,
		        metadata={"help": "Max input length for the source text"},
		    )
		    target_max_len: Optional[int] = field(
		        default=32,
		        metadata={"help": "Max input length for the target text"},
		    )


		def main():
		    

		    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

		    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(path_for_args))

		    if (
		        os.path.exists(training_args.output_dir)
		        and os.listdir(training_args.output_dir)
		        and training_args.do_train
		        and not training_args.overwrite_output_dir
		    ):
		        raise ValueError(
		            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
		        )

		    logging.basicConfig(
		        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		        datefmt="%m/%d/%Y %H:%M:%S",
		        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
		    )
		    logger.warning(
		        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		        training_args.local_rank,
		        training_args.device,
		        training_args.n_gpu,
		        bool(training_args.local_rank != -1),
		        training_args.fp16,
		    )
		    logger.info("Training/evaluation parameters %s", training_args)

		    set_seed(training_args.seed)

		    tokenizer = T5Tokenizer.from_pretrained(
		        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		        cache_dir=model_args.cache_dir,
		    )
		    model = T5ForConditionalGeneration.from_pretrained(
		        model_args.model_name_or_path,
		        cache_dir=model_args.cache_dir,
		    )

		    ############################################################

		    if frozen_layers > 0:

		        for param in model.encoder.embed_tokens.parameters():
		            param.requires_grad = False

		        layers_to_freeze = model.encoder.block[:frozen_layers]
		        for module in layers_to_freeze:
		            for param in module.parameters():
		                param.requires_grad = False

		    ############################################################



		    print('loading data')
		    train_dataset  = torch.load(data_args.train_file_path)
		    valid_dataset = torch.load(data_args.valid_file_path)
		    print('loading done')

		    trainer = Trainer(
		        model=model,
		        args=training_args,
		        train_dataset=train_dataset,
		        eval_dataset=valid_dataset,
		        data_collator=T2TDataCollator(),
		        #prediction_loss_only=True
		    )

		    if training_args.do_train:
		        trainer.train(
		            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
		        )
		        trainer.save_model()


		def _mp_fn(index):
		    # For xla_spawn (TPUs)
		    main()

		print("Beginning spawning")

		main()

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


		final_results = evaluate(references, predictions)

		print("----------------------------------------")
		print("Final results for " + str(learning_rate))
		print("----------------------------------------")
		print(final_results['exact_match'])
		print(final_results['f1'])
		print("----------------------------------------")

		exact_match_scores.append(final_results['exact_match'])
		f1_scores.append(final_results['f1'])

		results_string += "Exact Match: " + str(final_results['exact_match']) + "\n"
		results_string += "F1-Score: " + str(final_results['f1']) + "\n"


###############################################################


print("------------------------------------------")
print("Final Exact Match and F1 Averages for " + str(learning_rate) + " and " + str(frozen_layers) + " frozen layers.")
print(str(statistics.mean(exact_match_scores)))
print(str(statistics.mean(f1_scores)))
print("------------------------------------------")
print("Final Exact Match and F1 StDs")
print(str(statistics.stdev(exact_match_scores)))
print(str(statistics.stdev(f1_scores)))
print("------------------------------------------")


###############################################################


checkpoint_path = "GeneralQuestionAnswering_T5_" + model_choice.replace("/", "-") + "_" + str(learning_rate) + "_"
checkpoint_path += str(frozen_layers) + ".txt"

with open(checkpoint_path, "w") as text_file:
    text_file.write(results_string)

