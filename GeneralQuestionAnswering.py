
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, load_metric
from transformers import DefaultDataCollator, AutoTokenizer, get_scheduler, AutoModelForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm
import torch.nn as nn

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

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

################################################################

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

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

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

num_epochs = 10 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 3 #1 #5
frozen_choice = False
#chosen_learning_rate = 5e-6 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

validation_set_scoring = False

assigned_batch_size = 8
gradient_accumulation_multiplier = 4

validation_set_scoring = True

############################################################

warmup_steps_count = 2000
#learning_rate_choices = [0.0001, 1e-5, 2e-5, 5e-5, 5e-6]
learning_rate_choices = [1e-5]

model_choice = "distilbert-base-uncased"
checkpoint_path = 'checkpoints/experiment_QA_793.pt'
dataset_version = "./triviaqa_dataset_preprocessed_256_384_word_context_one_answer"

triviaqa_dataset = load_from_disk(dataset_version)
triviaqa_dataset.set_format("torch")

################################################################

tokenizer = AutoTokenizer.from_pretrained(model_choice)

################################################################

learning_rate_to_results_dict = {}

for chosen_learning_rate in learning_rate_choices:

	print("--------------------------------------------------------------------------")
	print("Starting new learning rate: " + str(chosen_learning_rate))
	print("--------------------------------------------------------------------------")

	execution_start = time.time()

	print("Model: " + model_choice)
	print("Dropout: " + str(current_dropout))
	print("Frozen Choice: " + str(frozen_choice))
	print("Number of Runs: " + str(number_of_runs))
	print('Learning Rate: ' + str(chosen_learning_rate))
	print("Checkpoint Path: " + checkpoint_path)
	print("Number of Frozen Layers: " + str(frozen_layers))
	print("Frozen Embeddings: " + str(frozen_embeddings))
	print("Patience: " + str(patience_value))
	print("Average Hidden Layers: " + str(average_hidden_state))
	print("Validation Set Choice: " + str(validation_set_scoring))
	print("Number of Epochs: " + str(num_epochs))
	print("Number of Warmup Steps: " + str(warmup_steps_count))
	print("Dataset Version: " + str(dataset_version))

	########################################################################

	model_choice = "distilbert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(model_choice)

	########################################################################

	micro_averages = []
	macro_averages = []
	inference_times = []

	for i in range(0, number_of_runs):

	    run_start = time.time()

	    print("Loading Model")

	    train_dataloader = DataLoader(triviaqa_dataset['train'], batch_size=assigned_batch_size)
	    validation_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)
	    eval_dataloader = DataLoader(triviaqa_dataset['test'], batch_size=assigned_batch_size)

	    print("Sizes of Training, Validation, and Test Sets")
	    print(len(triviaqa_dataset['train']))
	    print(len(triviaqa_dataset['validation']))
	    print(len(triviaqa_dataset['test']))

	    ############################################################

	    #model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
	    #						average_hidden_state, frozen_embeddings)

	    model = AutoModelForQuestionAnswering.from_pretrained(model_choice)

	    model.to(device)

	    ############################################################

	    criterion = nn.CrossEntropyLoss()
	    optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6

	    num_training_steps = num_epochs * len(train_dataloader)

	    lr_scheduler = get_scheduler(
	        name="linear", optimizer=optimizer, num_warmup_steps=warmup_steps_count, num_training_steps=num_training_steps
	    )

	    ############################################################



	    # to track the training loss as the model trains
	    train_losses = []
	    # to track the validation loss as the model trains
	    valid_losses = []
	    # to track the average training loss per epoch as the model trains
	    avg_train_losses = []
	    # to track the average validation loss per epoch as the model trains
	    avg_valid_losses = []


	    # import EarlyStopping
	    from pytorchtools import EarlyStopping
	    # initialize the early_stopping object
	    early_stopping = EarlyStopping(patience=patience_value, verbose=True, path=checkpoint_path)
	    #early_stopping = EarlyStopping(patience=10, verbose=True)

	    print("Checkpoint Path: " + checkpoint_path)


	    print("Beginning Training")

	    total_epochs_performed = 0

	    for epoch in range(num_epochs):

	        total_epochs_performed += 1

	        print("Current Epoch: " + str(epoch))

	        progress_bar = tqdm(range(len(train_dataloader)))

	        gradient_accumulation_count = 0

	        model.train()
	        for batch in train_dataloader:

	            #with torch.no_grad():

	                #dummy_start_positions = torch.Tensor([[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5]]).to(device)
	                #dummy_end_positions = torch.Tensor([[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5],[0, 5]]).to(device)

	                new_batch = {'input_ids': batch['input_ids'].to(device),
	                			 'attention_mask': batch['attention_mask'].to(device),
	                			 'start_positions': batch['start_positions'].to(device),
	                			 'end_positions': batch['end_positions'].to(device)}

	                outputs = model(**new_batch)

	                #print("model outputs")
	                #print(outputs.keys())
	                #print(outputs)

	                #print("Batch Shapes")
	                #print(new_batch['start_positions'])
	                #print(new_batch['end_positions'])

	                loss = outputs.loss

	                loss.backward()

	                gradient_accumulation_count += 1
	                if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
	                    optimizer.step()
	                    lr_scheduler.step()
	                    optimizer.zero_grad()
	                
	                progress_bar.update(1)
	                train_losses.append(loss.item())


	        progress_bar = tqdm(range(len(validation_dataloader)))

	        model.eval()
	        for batch in validation_dataloader:

	            #with torch.no_grad():
	            
	                new_batch = {'input_ids': batch['input_ids'].to(device),
	                			 'attention_mask': batch['attention_mask'].to(device),
	                			 'start_positions': batch['start_positions'].to(device),
	                			 'end_positions': batch['end_positions'].to(device)}
	                outputs = model(**new_batch)

	                loss = outputs.loss
	                progress_bar.update(1)

	                valid_losses.append(loss.item())


	        # print training/validation statistics 
	        # calculate average loss over an epoch
	        train_loss = np.average(train_losses)
	        valid_loss = np.average(valid_losses)
	        avg_train_losses.append(train_loss)
	        avg_valid_losses.append(valid_loss)
	        
	        epoch_len = len(str(num_epochs))
	        
	        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
	                     f'train_loss: {train_loss:.5f} ' +
	                     f'valid_loss: {valid_loss:.5f}')
	        
	        print(print_msg)
	        
	        # clear lists to track next epoch
	        train_losses = []
	        valid_losses = []
	        
	        # early_stopping needs the validation loss to check if it has decresed, 
	        # and if it has, it will make a checkpoint of the current model
	        early_stopping(valid_loss, model)
	        
	        if early_stopping.early_stop:
	            print("Early stopping")
	            break



	    ############################################################

	    print("Loading the Best Model")

	    model.load_state_dict(torch.load(checkpoint_path))



	    ############################################################

	    print("Beginning Evaluation")

	    metric = load_metric("accuracy")
	    #model.eval()

	    total_start_position_predictions = torch.FloatTensor([]).to(device)
	    total_start_position_references = torch.FloatTensor([]).to(device)

	    total_end_position_predictions = torch.FloatTensor([]).to(device)
	    total_end_position_references = torch.FloatTensor([]).to(device)

	    inference_start = time.time()

	    #progress_bar = tqdm(range(len(eval_dataloader)))
	    #for batch in eval_dataloader:

	    progress_bar = tqdm(range(len(eval_dataloader)))
	    for batch in eval_dataloader:

	        with torch.no_grad():

	            new_batch = {'input_ids': batch['input_ids'].to(device),
	                		 'attention_mask': batch['attention_mask'].to(device),
	                		 'start_positions': batch['start_positions'].to(device),
	                		 'end_positions': batch['end_positions'].to(device)}

	            outputs = model(**new_batch)

	            start_logits = outputs.start_logits
	            end_logits = outputs.end_logits

	            start_predictions = torch.argmax(start_logits, dim=-1)
	            end_predictions = torch.argmax(end_logits, dim=-1)

	            #print("start_predictions")
	            #print(start_predictions)
	            #print("end_predictions")
	            #print(end_predictions)

	            total_start_position_predictions = torch.cat((total_start_position_predictions, start_predictions), 0)
	            total_start_position_references = torch.cat((total_start_position_references, new_batch['start_positions']), 0)

	            total_end_position_predictions = torch.cat((total_end_position_predictions, end_predictions), 0)
	            total_end_position_references = torch.cat((total_end_position_references, new_batch['end_positions']), 0)

	            progress_bar.update(1)



	    inference_end = time.time()
	    total_inference_time = inference_end - inference_start
	    inference_times.append(total_inference_time)

	    ############################################################

	    print("--------------------------")
	    print("Predictions Shapes")
	    print(total_start_position_predictions.shape)
	    print(total_start_position_references.shape)
	    print(total_end_position_predictions.shape)
	    print(total_end_position_references.shape)

	    results = metric.compute(references=total_start_position_references, predictions=total_start_position_predictions)
	    print("Accuracy for Test Set: " + str(results['accuracy']))

	    f_1_metric = load_metric("f1")
	    macro_f_1_results = f_1_metric.compute(average='macro', references=total_start_position_references, predictions=total_start_position_predictions)
	    print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
	    micro_f_1_results = f_1_metric.compute(average='micro', references=total_start_position_references, predictions=total_start_position_predictions)
	    print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))

	    micro_averages.append(micro_f_1_results['f1'] * 100)
	    macro_averages.append(macro_f_1_results['f1'] * 100)


	print("Processing " + dataset_version + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout")
	print('micro_averages: ' + str(micro_averages))
	print("Micro F1 Average: " + str(statistics.mean(micro_averages)))
	if len(micro_averages) > 1:
	    print("Micro F1 Standard Variation: " + str(statistics.stdev(micro_averages)))

	print('macro_averages: ' + str(macro_averages))
	print("Macro F1 Average: " + str(statistics.mean(macro_averages)))
	if len(macro_averages) > 1:
	    print("Macro F1 Standard Variation: " + str(statistics.stdev(macro_averages)))

	print("Inference Time Average: " + str(statistics.mean(inference_times)))
	print("Dataset Execution Run Time: " + str((time.time() - execution_start) / number_of_runs))
	print("Epoch Average Time: " + str((time.time() - run_start) / total_epochs_performed))

	

	############################################################

	current_learning_rate_results[dataset + "_micro_f1_average"] =  statistics.mean(micro_averages)
	if len(micro_averages) > 1:
	    current_learning_rate_results[dataset + "_micro_f1_std"] =  statistics.stdev(micro_averages)
	current_learning_rate_results[dataset + "_macro_f1_average"] =  statistics.mean(macro_averages)
	if len(macro_averages) > 1:
	    current_learning_rate_results[dataset + "_macro_f1_std"] =  statistics.stdev(macro_averages)

	############################################################

learning_rate_to_results_dict[str(chosen_learning_rate)] = current_learning_rate_results




