

import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, load_metric
from transformers import DefaultDataCollator, AutoTokenizer, get_scheduler, AutoModel
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm
import torch.nn as nn
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

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings):

          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice == "roberta-large":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 384
            self.encoderModel = model_encoding

          elif model_choice == "t5-small":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 512
            self.encoderModel = model_encoding

          elif model_choice == "microsoft/deberta-v2-xlarge":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True)
            embedding_size = 1536
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

            if model_choice == "distilbert-base-uncased":

                print("Number of Layers: " + str(len(list(self.encoderModel.transformer.layer))))

                layers_to_freeze = self.encoderModel.transformer.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            elif model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            elif model_choice == 't5-base':

                layers_to_freeze = self.encoderModel.encoder.block[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

                layers_to_freeze = self.encoderModel.decoder.block[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False



          
          if frozen_embeddings == True:
            
            print("Frozen Embeddings Layer")
            
            if model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':
                for param in self.encoderModel.embeddings.parameters():
                    param.requires_grad = False

            elif model_choice == 't5-base':

                for param in self.encoderModel.shared.parameters():
                    param.requires_grad = False

                for param in self.encoderModel.encoder.embed_tokens.parameters():
                    param.requires_grad = False

                for param in self.encoderModel.decoder.embed_tokens.parameters():
                    param.requires_grad = False

            else:
                for param in self.encoderModel.embeddings.parameters():
                    param.requires_grad = False


          


          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state

          ############################################################################

          if model_choice == "allenai/scibert_scivocab_uncased":

            self.classifier = nn.Sequential(
              								nn.Linear(in_features=embedding_size, out_features=2, bias=True)
              							 )

          elif model_choice == "roberta-large":

          	self.classifier = nn.Sequential(
              								nn.Linear(in_features=embedding_size, out_features=2, bias=True)
              							 )

          else:

          	self.classifier = nn.Sequential(
              								nn.Linear(in_features=embedding_size, out_features=2, bias=True)
              							 )



          

    def forward(self, input_ids, attention_mask, start_positions, end_positions, decoded_inputs=None, token_type_ids=None):

        if model_choice == 't5-base' or model_choice == 't5-small':

            if decoded_inputs == None:
                print("Error with decoded_inputs!")

            output_hidden_states = self.encoderModel(input_ids=input_ids, decoder_input_ids=decoded_inputs)#['last_hidden_state']
            last_hidden_state = output_hidden_states['last_hidden_state']

        else:

            output_hidden_states = self.encoderModel(input_ids, attention_mask)['hidden_states']#['last_hidden_state']
            last_hidden_state = output_hidden_states[len(output_hidden_states) - 1]

	    ##################################################################

        classifier_output = self.classifier(last_hidden_state)
        start_logits = classifier_output[:, :, 0]
        end_logits = classifier_output[:, :, 1]

        return {'start_logits': start_logits, 'end_logits': end_logits}

##################################################

def compute_f1(predictions_list, references_list):

	f1_scores = []

	for index in range(0, len(predictions_list)):

	    prediction = predictions_list[index]
	    truth = references_list[index]

	    pred_tokens = [x for x in range(prediction[0], prediction[1] + 1)]
	    truth_tokens = [x for x in range(truth[0], truth[1] + 1)]

	    common_tokens = set(pred_tokens) & set(truth_tokens)
	    
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

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

num_epochs = 10 #1000 #10
patience_value = 3 #10 #3
current_dropout = True
number_of_runs = 3 #1 #5
frozen_choice = False
average_hidden_state = False

assigned_batch_size = 2
gradient_accumulation_multiplier = 16

validation_set_scoring = False

warmup_steps_count_ratio = 0.2

############################################################

frozen_layers = 6 # For freezing k-later layers of transformer model
frozen_embeddings = True # For freezing input embeddings layer of transformer model

model_choice = "bert-base-uncased"

chosen_checkpoint_path = "paper_results_qa/bert-base-uncased/5e-05_6_True_3False_Run_0.pt"

half_configuration = True

chosen_dataset = 'trivia_qa'

############################################################

context_cutoff_count = 1024
context_token_count = 512
multi_answer = False
remove_missing_answers = False

reduced_sample = False

############################################################

dataset_version = "./" + chosen_dataset + "_dataset_" + model_choice + "_" + str(context_cutoff_count) + "_" + str(context_token_count)
dataset_version += "_" + str(multi_answer) + "_" + str(remove_missing_answers) + "_" + str(reduced_sample)

triviaqa_dataset = load_from_disk(dataset_version)
triviaqa_dataset.set_format("torch")

################################################################

run_start = time.time()

train_dataloader = DataLoader(triviaqa_dataset['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(triviaqa_dataset['test'], batch_size=assigned_batch_size)

print("Sizes of Training, Validation, and Test Sets")
print(len(triviaqa_dataset['train']))
print(len(triviaqa_dataset['validation']))
print(len(triviaqa_dataset['test']))

############################################################

model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
						average_hidden_state, frozen_embeddings)

model.to(device)

print("Loading the Best Model")

model.load_state_dict(torch.load(chosen_checkpoint_path))

if half_configuration == True:
	print("Using Half Precision!")
	model.half()

############################################################

print("Beginning Evaluation")

metric = load_metric("accuracy")
#model.eval()

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

print("Exact Match: " + str(exact_match_score * 100))
print("F1-Score: " + str(f1_score * 100))



