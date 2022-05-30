
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import DefaultDataCollator, AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import random
import torch
import os

################################################################

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

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

	    current_start_positions = []
	    current_end_positions = []

	    for j in range(0, len(answers[i]['text'])):

	        answer = {'text': [answers[i]['text'][j]], "answer_start": [answers[i]['answer_start'][j]]}

	        #answer = answers[i]
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
	            current_start_positions.append(0)
	            current_end_positions.append(0)
	        else:
	            # Otherwise it's the start and end token positions
	            idx = context_start
	            while idx <= context_end and offset[idx][0] <= start_char:
	                idx += 1
	            current_start_positions.append(idx - 1)

	            idx = context_end
	            while idx >= context_start and offset[idx][1] >= end_char:
	                idx -= 1
	            current_end_positions.append(idx + 1)

	    start_positions.append(current_start_positions)
	    end_positions.append(current_end_positions)

	inputs["start_positions"] = start_positions
	inputs["end_positions"] = end_positions
	return inputs

########################################################################

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def findIndexesOfAnswers(substring_list, context):

	found_aliases = []
	indices = []

	for substring in substring_list:
		
		currentIndices = list(find_all(context, substring))

		for currentIndex in currentIndices:
			if currentIndex >= 0:
				found_aliases.append(substring)
				indices.append(currentIndex)

	return found_aliases, indices

########################################################################

def reformat_trivia_qa(examples):

	ids = []
	titles = []
	contexts = []
	questions = []
	answers = []

	for i in range(0, len(examples['question_id'])):

		current_context = (" ").join(examples['search_results'][i]['search_context'])

		if len(current_context) == 0:
			#print("Error!")
			current_context = (" ").join(examples['entity_pages'][i]['wiki_context'])
			if len(current_context) == 0:
				print("Major Error!")

		current_text, current_found_answers = findIndexesOfAnswers(examples['answer'][i]['aliases'], current_context)
		current_answers = {'text': current_text, 'answer_start': current_found_answers}

		if len(current_text) == 0:
			current_context = "Error"

		ids.append(examples['question_id'][i])
		titles.append("")
		contexts.append(current_context)
		questions.append(examples['question'][i])
		answers.append(current_answers)

	#################################################

	inputs = {
		'id': ids,
		'title': titles,
		'context': contexts,
		'question': questions,
		'answers': answers
	}

	return inputs


########################################################################


triviaqa_dataset = load_dataset("trivia_qa", 'rc')
squad = load_dataset('squad')

model_choice = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_choice)

########################################################################

triviaqa_dataset = triviaqa_dataset.map(reformat_trivia_qa, batched=True)

print("---------------------------------------------------------------------")
print("Before: " + str(len(triviaqa_dataset['train'])))
triviaqa_dataset['train'] = triviaqa_dataset['train'].filter(lambda x: x['context'] != "Error")
triviaqa_dataset['validation'] = triviaqa_dataset['validation'].filter(lambda x: x['context'] != "Error")
triviaqa_dataset['test'] = triviaqa_dataset['test'].filter(lambda x: x['context'] != "Error")
print("After: " + str(len(triviaqa_dataset['train'])))
print("---------------------------------------------------------------------")

triviaqa_dataset = triviaqa_dataset.map(preprocess_function, batched=True, remove_columns=triviaqa_dataset["train"].column_names)

########################################################################

triviaqa_dataset.set_format("torch")
triviaqa_dataset.save_to_disk("./triviaqa_dataset_preprocessed_v2")











