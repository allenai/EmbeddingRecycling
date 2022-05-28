
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import DefaultDataCollator, AutoTokenizer, get_scheduler
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

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings):

          super(CustomBERTModel, self).__init__()

          if model_choice == 'roberta-large':

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == 'distilbert-base-uncased':

            model_encoding = DistilBertModel.from_pretrained('distilbert-base-uncased')
            embedding_size = 768
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 768
            self.encoderModel = model_encoding


          if frozen == True:
            print("Freezing the model parameters")
            for param in self.encoderModel.parameters():
                param.requires_grad = False

          if frozen_layer_count > 0:

            if model_choice == "t5-3b":

                print("Freezing T5-3b")
                print("Number of Layers: " + str(len(self.encoderModel.encoder.block)))

                for parameter in self.encoderModel.parameters():
                    parameter.requires_grad = False

                for i, m in enumerate(self.encoderModel.encoder.block):        
                    #Only un-freeze the last n transformer blocks
                    if i+1 > 24 - frozen_layer_count:
                        print(str(i) + " Layer")
                        for parameter in m.parameters():
                            parameter.requires_grad = True

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

          
          if frozen_embeddings == True:
            print("Frozen Embeddings Layer")
            for param in self.encoderModel.embeddings.parameters():
                param.requires_grad = False



          ##################################################################

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state


          

    def forward(self, input_ids, attention_mask):

        model_output = self.encoderModel(input_ids, attention_mask)

        return model_output



############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

num_epochs = 100 #1000 #10
patience_value = 10 #10 #3
current_dropout = True
number_of_runs = 10 #1 #5
frozen_choice = False
#chosen_learning_rate = 5e-6 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 3 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = True
average_hidden_state = False

validation_set_scoring = False

assigned_batch_size = 8
gradient_accumulation_multiplier = 4

learning_rate_choices = [0.0001, 1e-5, 2e-5, 5e-5, 5e-6]

validation_set_scoring = True

############################################################

dataset = "wikipedia"
model_choice = "distilbert-base-uncased"
checkpoint_path = 'checkpoints/experiment10_768.pt'

reformatted_examples_train = reformatExamples('../triviaqa/squad_format_for_triviaqa_qa_' + dataset + "-train")
reformatted_examples_dev = reformatExamples('../triviaqa/squad_format_for_triviaqa_qa_' + dataset + "-dev")

################################################################

tokenizer = AutoTokenizer.from_pretrained(model_choice)

################################################################

learning_rate_to_results_dict = {}

for chosen_learning_rate in learning_rate_choices:

	print("--------------------------------------------------------------------------")
	print("Starting new learning rate: " + str(chosen_learning_rate))
	print("--------------------------------------------------------------------------")

	execution_start = time.time()

	print("Dataset: " + dataset)
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

	########################################################

	qa_dataset = DatasetDict({'train' : training_dataset_arrow, 
	                          'validation': validation_dataset_arrow, 
	                          'test' : test_dataset_arrow})
	        
	tokenized_datasets = qa_dataset.map(preprocess_function, batched=True)

	########################################################

	model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

	############################################################

	micro_averages = []
	macro_averages = []
	inference_times = []

	for i in range(0, number_of_runs):

	    run_start = time.time()

	    print("Loading Model")

	    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
	    validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
	    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

	    ############################################################

	    model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
	    						average_hidden_state, frozen_embeddings)

	    model.to(device)

	    ############################################################

	    criterion = nn.CrossEntropyLoss()
	    optimizer = Adam(model.parameters(), lr=chosen_learning_rate) #5e-6

	    num_training_steps = num_epochs * len(train_dataloader)

	    lr_scheduler = get_scheduler(
	        name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
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
	            
	                #batch = {k: v.to(device) for k, v in batch.items()}
	                labels = batch['labels'].to(device)

	                new_batch = {'input_ids': batch['input_ids'].to(device), 
	                			 'attention_mask': batch['attention_mask'].to(device)}
	                outputs = model(**new_batch)

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
	            
	                #batch = {k: v.to(device) for k, v in batch.items()}
	                labels = batch['labels'].to(device)

	                new_batch = {'input_ids': batch['input_ids'].to(device), 
	                			 'attention_mask': batch['attention_mask'].to(device)}
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

	    total_predictions = torch.FloatTensor([]).to(device)
	    total_references = torch.FloatTensor([]).to(device)

	    inference_start = time.time()

	    #progress_bar = tqdm(range(len(eval_dataloader)))
	    #for batch in eval_dataloader:

	    progress_bar = tqdm(range(len(eval_dataloader)))
	    for batch in eval_dataloader:

	        with torch.no_grad():

	            #batch = {k: v.to(device) for k, v in batch.items()}
	            labels = batch['labels'].to(device)

	            new_batch = {'input_ids': batch['input_ids'].to(device), 
	            			 'attention_mask': batch['attention_mask'].to(device)}

	            outputs = model(**new_batch)

	            logits = outputs.logits
	            predictions = torch.argmax(logits, dim=-1)
	            metric.add_batch(predictions=predictions, references=labels)

	            total_predictions = torch.cat((total_predictions, predictions), 0)
	            total_references = torch.cat((total_references, labels), 0)

	            progress_bar.update(1)



	    inference_end = time.time()
	    total_inference_time = inference_end - inference_start
	    inference_times.append(total_inference_time)

	    ############################################################

	    print("--------------------------")
	    print("Predictions Shapes")
	    print(total_predictions.shape)
	    print(total_references.shape)

	    results = metric.compute(references=total_predictions, predictions=total_references)
	    print("Accuracy for Test Set: " + str(results['accuracy']))

	    f_1_metric = load_metric("f1")
	    macro_f_1_results = f_1_metric.compute(average='macro', references=total_predictions, predictions=total_references)
	    print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
	    micro_f_1_results = f_1_metric.compute(average='micro', references=total_predictions, predictions=total_references)
	    print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))

	    micro_averages.append(micro_f_1_results['f1'] * 100)
	    macro_averages.append(macro_f_1_results['f1'] * 100)


	print("Processing " + dataset + " using " + model_choice + " with " + str(current_dropout) + " for current_dropout")
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




