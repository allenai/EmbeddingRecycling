

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


#import torch
#import torch_xla
#import torch_xla.core.xla_model as xm

import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm
from opendelta import AdapterModel, BitFitModel

###############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

#model_choice = "t5-base"
model_choice = "google/t5-large-lm-adapt"

bottleneck_value = 256
use_all_adapter = True

learning_rate_choices = [5e-5, 5e-6]

###############################################################

if model_choice in ["google/t5-large-lm-adapt"]:

	unfrozen_components = ['lm_head']
	#unfrozen_components = []
	
	starting_layer_for_adapters = 12
	if use_all_adapter == True:
		starting_layer_for_adapters = 0

	#unfrozen_components.append('encoder')

	for i in range(starting_layer_for_adapters, 24):
		attention_adapter = 'encoder.block.' + str(i) + ".layer.0.adapter"
		output_adapter = 'encoder.block.' + str(i) + ".layer.1.adapter"
		unfrozen_components.append(attention_adapter)
		unfrozen_components.append(output_adapter)

	for i in range(0, 24):
		attention_adapter = 'decoder.block.' + str(i) + ".layer.0.adapter"
		output_adapter = 'decoder.block.' + str(i) + ".layer.2.adapter"
		unfrozen_components.append(attention_adapter)
		unfrozen_components.append(output_adapter)


############################################################

for learning_rate in learning_rate_choices:

	print("-------------------------------------------------------------")
	print("Chosen Model: " + str(model_choice))
	print("Learning Rate: " + str(learning_rate))
	print("Starting Layer of Adapters: " + str(starting_layer_for_adapters))
	print("-------------------------------------------------------------")

	output_directory = "models/" + model_choice + "/" + str(learning_rate) + "/" + str(use_all_adapter)
	path_for_args = str(model_choice.replace("/", "-")) + "_" + str(use_all_adapter) + "_" + str(learning_rate) + "_args.json"

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
	  "cache_dir": output_directory
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
	train_dataset = train_dataset.map(add_eos_to_examples)
	# map convert_to_features batch wise
	train_dataset = train_dataset.map(convert_to_features, batched=True)

	valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
	valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

	non_torch_valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
	non_torch_valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


	# set the tensor type and the columns which the dataset should return
	columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
	train_dataset.set_format(type='torch', columns=columns)
	valid_dataset.set_format(type='torch', columns=columns)

	len(train_dataset), len(valid_dataset)

	# cach the dataset, so we can load it directly for training

	torch.save(train_dataset, 'train_data.pt')
	torch.save(valid_dataset, 'valid_data.pt')

	print("Finishing Preprocessing")

	########################################################################################

	logger = logging.getLogger(__name__)

	# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
	# this is necessacry because the trainer directly passes this dict as arguments to the model
	# so make sure the keys match the parameter names of the forward method
	@dataclass
	class T2TDataCollator():
	    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
	        """
	        Take a list of samples from a Dataset and collate them into a batch.
	        Returns:
	            A dictionary of tensors
	        """

	        #print("Given batch")
	        #print(batch)


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
	    """
	    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	    """

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
	    """
	    Arguments pertaining to what data we are going to input our model for training and eval.
	    """
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
	    # See all possible arguments in src/transformers/training_args.py
	    # or by passing the --help flag to this script.
	    # We now keep distinct sets of args, for a cleaner separation of concerns.

	    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

	    # we will load the arguments from a json file, 
	    #make sure you save the arguments in at ./args.json
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

	    # Setup logging
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

	    # Set seed
	    set_seed(training_args.seed)

	    # Load pretrained model and tokenizer
	    #
	    # Distributed training:
	    # The .from_pretrained methods guarantee that only one local process can concurrently
	    # download model & vocab.

	    tokenizer = T5Tokenizer.from_pretrained(
	        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
	        cache_dir=model_args.cache_dir,
	    )
	    model = AutoModelForSeq2SeqLM.from_pretrained(
	        model_args.model_name_or_path,
	        cache_dir=model_args.cache_dir,
	    )

	    ############################################################

	    delta_model = AdapterModel(backbone_model=model, bottleneck_dim=bottleneck_value)
	    delta_model.freeze_module(exclude=unfrozen_components, set_state_dict=True)
	    delta_model.log()

	    ############################################################



	    # Get datasets
	    print('loading data')
	    train_dataset  = torch.load(data_args.train_file_path)
	    valid_dataset = torch.load(data_args.valid_file_path)
	    print('loading done')

	    #print("Loaded datasets")
	    #print(train_dataset)
	    #print(valid_dataset)

	    # Initialize our Trainer
	    trainer = Trainer(
	        model=model,
	        args=training_args,
	        train_dataset=train_dataset,
	        eval_dataset=valid_dataset,
	        data_collator=T2TDataCollator(),
	        #prediction_loss_only=True
	    )

	    # Training
	    if training_args.do_train:
	        trainer.train(
	            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
	        )
	        trainer.save_model()
	        # For convenience, we also re-save the tokenizer to the same directory,
	        # so that you can share your model easily on huggingface.co/models =)
	        #if trainer.is_world_master():
	        #    tokenizer.save_pretrained(training_args.output_dir)

	    # Evaluation
	    results = {}
	    if training_args.do_eval and training_args.local_rank in [-1, 0]:
	        logger.info("*** Evaluate ***")

	        eval_output = trainer.evaluate()

	        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
	        with open(output_eval_file, "w") as writer:
	            logger.info("***** Eval results *****")
	            for key in sorted(eval_output.keys()):
	                logger.info("  %s = %s", key, str(eval_output[key]))
	                writer.write("%s = %s\n" % (key, str(eval_output[key])))
	    
	        results.update(eval_output)
	    
	    return results


	def _mp_fn(index):
	    # For xla_spawn (TPUs)
	    main()

	"""## Train"""

	"""Let's write the arguments in a dict and store in a json file. The above code will load this file and parse the arguments."""

	"""Start training!"""

	#import torch_xla.distributed.xla_multiprocessing as xmp
	#xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')

	print("Beginning spawning")

	main()

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


	final_results = evaluate(references, predictions)

	print("----------------------------------------")
	print("Final results for " + str(learning_rate))
	print("----------------------------------------")
	print(final_results['exact_match'])
	print(final_results['f1'])
	print("----------------------------------------")

