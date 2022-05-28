
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

def reformatExamples(file_name):

	reformatted_examples = []
	with open('../triviaqa/triviaqa_squad_format.json', 'r') as f:
		data = json.load(f)

		for item in tqdm(data):

			for sub_item in item['paragraphs']:

				if len(sub_item['qas'][0]['answers']) > 0:

					reformatted_answers = {"text": [sub_item['qas'][0]['answers'][0]['text']],
										   "answer_start": [sub_item['qas'][0]['answers'][0]['answer_start']]}

					current_example = 	{
										   'id': sub_item['qas'][0]['id'],
										   'title': sub_item['qas'][0]['question'],
										   'context': sub_item['context'],
										   'question': sub_item['qas'][0]['question'],
										   'answers': reformatted_answers
										}
					reformatted_examples.append(current_example)

				else:

					print("Missing answer!")

	return reformatted_examples

########################################################################






validation_set_scoring = True
chosen_dataset = "wikipedia"

reformatted_examples_train = reformatExamples('../triviaqa/squad_format_for_triviaqa_qa_' + chosen_dataset + "-train")
reformatted_examples_dev = reformatExamples('../triviaqa/squad_format_for_triviaqa_qa_' + chosen_dataset + "-dev")

################################################################

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


################################################################

if validation_set_scoring == True:

    training_df = pd.DataFrame(reformatted_examples_train)
    train, validation = train_test_split(training_df, test_size=0.15, shuffle=True, random_state=random_state)
    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)

    training_dataset_pandas = train#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = Dataset(training_dataset_arrow)

    validation_dataset_pandas = validation#[:1000]
    validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
    validation_dataset_arrow = Dataset(validation_dataset_arrow)

    test_dataset_pandas = pd.DataFrame(reformatted_examples_dev)
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = Dataset(test_dataset_arrow)

else:

    training_dataset_pandas = pd.DataFrame(reformatted_examples_train)#[:1000]
    training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
    training_dataset_arrow = Dataset(training_dataset_arrow)

    validation_dataset_pandas = pd.DataFrame(reformatted_examples_dev)#[:1000]
    validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
    validation_dataset_arrow = Dataset(validation_dataset_arrow)

    test_dataset_pandas = pd.DataFrame(reformatted_examples_test)
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = Dataset(test_dataset_arrow)

################################################################

qa_dataset = DatasetDict({'train' : training_dataset_arrow, 
                          'validation': validation_dataset_arrow, 
                          'test' : test_dataset_arrow})
        
tokenized_datasets = qa_dataset.map(preprocess_function, batched=True)

#tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

################################################################

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
