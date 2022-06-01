
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import DefaultDataCollator, AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm

from urllib.request import urlopen, Request

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

def preprocess_function_multi_answer(examples):
	questions = [q.strip() for q in examples["question"]]
	inputs = tokenizer(
		questions,
		examples["context"],
		max_length=context_token_count,
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

	    ################################################

	    while len(current_start_positions) < 32:
	    	current_start_positions.append(-1)
	    	current_end_positions.append(-1)

	    ################################################

	    start_positions.append(current_start_positions)
	    end_positions.append(current_end_positions)

	inputs["start_positions"] = start_positions
	inputs["end_positions"] = end_positions
	return inputs

########################################################################

def preprocess_function_single_answer(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=context_token_count,
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

	lowestIndex = float("inf")
	lowestAlias = ""

	for substring in substring_list:
		
		currentIndices = list(find_all(context, substring))

		for currentIndex in currentIndices:
			if currentIndex >= 0:

				if currentIndex < lowestIndex:

					lowestAlias = substring
					lowestIndex = currentIndex

				found_aliases.append(substring)
				indices.append(currentIndex)

	###############################

	if multi_answer == True:

		return found_aliases, indices

	else:

		if len(found_aliases) > 0: 
			return [lowestAlias], [lowestIndex]
		else:
			return [], []


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

		current_context = (" ").join(current_context.split(" ")[:context_cutoff_count])

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

model_choice = 'roberta-large'#'allenai/scibert_scivocab_uncased'#'roberta-large'# "distilbert-base-uncased" 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_choice)

context_cutoff_count = 512
context_token_count = 256

multi_answer = False

chosen_dataset = "trivia_qa"
#chosen_dataset = "natural_questions"
#squad = load_dataset('squad')

#################################################################

if chosen_dataset == "trivia_qa":
	current_dataset = load_dataset(chosen_dataset, 'rc')
else:
	current_dataset = load_dataset(chosen_dataset)

save_path = "./" + chosen_dataset + "_dataset_" + model_choice + "_" + str(context_cutoff_count) + "_" + str(context_token_count)
save_path += "_" + str(multi_answer)

print("Preparing " + save_path)



#print("NQ Example")
#print(current_dataset['train'][0])











########################################################################

current_dataset = current_dataset.map(reformat_trivia_qa, batched=True)

####################################################################

train, validation = train_test_split(current_dataset['train'], test_size=0.15, shuffle=True, random_state=random_state)

training_dataset_pandas = pd.DataFrame(train)
training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
training_dataset_arrow = Dataset(training_dataset_arrow)
current_dataset['train'] = training_dataset_arrow

current_dataset['test'] = current_dataset['validation']

validation_dataset_pandas = pd.DataFrame(validation)#[:1000]
validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
validation_dataset_arrow = Dataset(validation_dataset_arrow)
current_dataset['validation'] = validation_dataset_arrow

print("current_dataset['train'] after")
print(type(current_dataset['train']))
print(type(current_dataset['validation']))
print(type(current_dataset['test']))

####################################################################

print("---------------------------------------------------------------------")
print("Before: " + str(len(current_dataset['train'])))
print("Before: " + str(len(current_dataset['validation'])))
print("Before: " + str(len(current_dataset['test'])))
current_dataset['train'] = current_dataset['train'].filter(lambda x: x['context'] != "Error")
current_dataset['validation'] = current_dataset['validation'].filter(lambda x: x['context'] != "Error")
current_dataset['test'] = current_dataset['test'].filter(lambda x: x['context'] != "Error")
print("After: " + str(len(current_dataset['train'])))
print("After: " + str(len(current_dataset['validation'])))
print("After: " + str(len(current_dataset['test'])))
print("---------------------------------------------------------------------")

####################################################################

#print("A few examples")
#print(current_dataset['train'][1000])

if multi_answer == True:
	current_dataset = current_dataset.map(preprocess_function_multi_answer, batched=True, remove_columns=current_dataset["train"].column_names)
else:
	current_dataset = current_dataset.map(preprocess_function_single_answer, batched=True, remove_columns=current_dataset["train"].column_names)

####################################################################

print("---------------------------------------------------------------------")
print("Before: " + str(len(current_dataset['train'])))

if multi_answer == True:
	
	current_dataset['train'] = current_dataset['train'].filter(lambda x: 0 not in x['start_positions'] or 0 not in x['end_positions'])
	current_dataset['validation'] = current_dataset['validation'].filter(lambda x: 0 not in x['start_positions'] or 0 not in x['end_positions'])
	current_dataset['test'] = current_dataset['test'].filter(lambda x: 0 not in x['start_positions'] or 0 not in x['end_positions'])

	print("Multi Answer example")
	print(current_dataset['train'][10])

else:

	current_dataset['train'] = current_dataset['train'].filter(lambda x: x['start_positions'] != 0 or x['end_positions'] != 0)
	current_dataset['validation'] = current_dataset['validation'].filter(lambda x: x['start_positions'] != 0 or x['end_positions'] != 0)
	current_dataset['test'] = current_dataset['test'].filter(lambda x: x['start_positions'] != 0 or x['end_positions'] != 0)

print("After: " + str(len(current_dataset['train'])))
print("---------------------------------------------------------------------")

########################################################################

current_dataset.save_to_disk(save_path)











