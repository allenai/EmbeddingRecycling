

import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, load_metric
from transformers import DefaultDataCollator, AutoTokenizer, get_scheduler, AutoModel
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm import tqdm
import torch.nn as nn
from opendelta import AdapterModel, BitFitModel
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

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True, output_attentions=True)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice == "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True, output_attentions=True)
            embedding_size = 384
            self.encoderModel = model_encoding

          elif model_choice == "t5-small":

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True, output_attentions=True)
            embedding_size = 512
            self.encoderModel = model_encoding

          else:

            model_encoding = AutoModel.from_pretrained(model_choice, output_hidden_states=True, output_attentions=True)
            embedding_size = 768
            self.encoderModel = model_encoding



          if frozen == True:
            print("Freezing the model parameters")
            for param in self.encoderModel.parameters():
                param.requires_grad = False



          if frozen_layer_count > 0:

            if model_choice == "distilbert-base-uncased":

                #print(self.encoderModel.__dict__)
                print("Number of Layers: " + str(len(list(self.encoderModel.transformer.layer))))

                layers_to_freeze = self.encoderModel.transformer.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            elif model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':

                print("Number of Layers: " + str(len(list(self.encoderModel.roberta.encoder.layer))))

                layers_to_freeze = self.encoderModel.roberta.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            elif model_choice == 't5-base' or model_choice == 't5-small':

                layers_to_freeze = self.encoderModel.encoder.block[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

                #layers_to_freeze = self.encoderModel.decoder.block[:frozen_layer_count]
                #for module in layers_to_freeze:
                #    for param in module.parameters():
                #        param.requires_grad = False

            else:

                #print(self.encoderModel.__dict__)

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False



          
          if frozen_embeddings == True:
            print("Frozen Embeddings Layer")
            #print(self.encoderModel.__dict__)
            if model_choice == 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large':
                for param in self.encoderModel.roberta.embeddings.parameters():
                    param.requires_grad = False

            elif model_choice == 't5-base' or model_choice == 't5-small':

                for param in self.encoderModel.shared.parameters():
                    param.requires_grad = False

                for param in self.encoderModel.encoder.embed_tokens.parameters():
                    param.requires_grad = False

                #for param in self.encoderModel.decoder.embed_tokens.parameters():
                #    param.requires_grad = False

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



          

    def forward(self, input_ids, attention_mask, start_positions, end_positions, decoded_inputs=None, token_type_ids=None, answer_lengths=None):

        if model_choice == 't5-base' or model_choice == 't5-small':

            if decoded_inputs == None:
                print("Error with decoded_inputs!")

            output_hidden_states = self.encoderModel(input_ids=input_ids, decoder_input_ids=decoded_inputs)#['last_hidden_state']
            last_hidden_state = output_hidden_states['last_hidden_state']

        else:

            output = self.encoderModel(input_ids, attention_mask)#['last_hidden_state']
            last_hidden_state = output['hidden_states'][len(output['hidden_states']) - 1]

	    ##################################################################

        classifier_output = self.classifier(last_hidden_state)
        start_logits = classifier_output[:, :, 0]
        end_logits = classifier_output[:, :, 1]

	    #print("final output")
	    #print(classifier_output.shape)
	    #print(start_logits.shape)
	    #print(end_logits.shape)
				
        return {'start_logits': start_logits, 'end_logits': end_logits, 
        		'hidden_states': output['hidden_states'], 'attentions': output['attentions']}

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

num_epochs = 50 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 10 #1 #5
frozen_choice = False
#chosen_learning_rate = 5e-6 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

assigned_batch_size = 8
gradient_accumulation_multiplier = 4

validation_set_scoring = False

############################################################

model_choice = 'roberta-large'

############################################################

chosen_dataset = 'squad'
context_cutoff_count = 1024
context_token_count = 512
multi_answer = False
remove_missing_answers = False

reduced_sample = False

dataset_version = "./" + chosen_dataset + "_dataset_" + model_choice + "_" + str(context_cutoff_count) + "_" + str(context_token_count)
dataset_version += "_" + str(multi_answer) + "_" + str(remove_missing_answers) + "_" + str(reduced_sample)

triviaqa_dataset = load_from_disk(dataset_version)
triviaqa_dataset.set_format("torch")

train_dataloader = DataLoader(triviaqa_dataset['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)

############################################################

question_dataset = load_from_disk('./squad_dataset_roberta-large_1024_512_False_False_False_True')
question_dataset.set_format("torch")

question_dataloader = DataLoader(question_dataset['validation'], batch_size=assigned_batch_size)

############################################################

chosen_learning_rate = 5e-6

fully_finetuned_model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
	    								average_hidden_state, frozen_embeddings)

fully_finetuned_checkpoint_path = "paper_results_qa/" + model_choice.replace("/", "-") + "/" + str(chosen_learning_rate) + "_"
fully_finetuned_checkpoint_path += str(frozen_layers) + "_" + str(frozen_embeddings) + "_" + str(number_of_runs)
fully_finetuned_checkpoint_path += str(validation_set_scoring) + "_Run_" + str(0) + ".pt"

fully_finetuned_model.load_state_dict(torch.load(fully_finetuned_checkpoint_path))

fully_finetuned_model.to(device)

############################################################

frozen_layers = 12
frozen_embeddings = True

semifrozen_model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
	    						   average_hidden_state, frozen_embeddings)

semifrozen_checkpoint_path = "semifrozen_roberta_checkpoint"
semifrozen_model.load_state_dict(torch.load(semifrozen_checkpoint_path))

semifrozen_model.to(device)

############################################################

print("Beginning Evaluation")

metric = load_metric("accuracy")

total_start_position_predictions = torch.LongTensor([]).to(device)
total_start_position_references = torch.LongTensor([]).to(device)

total_end_position_predictions = torch.LongTensor([]).to(device)
total_end_position_references = torch.LongTensor([]).to(device)

inference_start = time.time()

finetuned_correct_answer_context_lengths = []
finetuned_wrong_answer_context_lengths = []

semifrozen_correct_answer_context_lengths = []
semifrozen_wrong_answer_context_lengths = []

progress_bar = tqdm(range(len(eval_dataloader)))
for batch, question_batch in zip(eval_dataloader, question_dataloader):

    with torch.no_grad():

        question_lengths = question_batch['question_lengths']

        new_batch = {k: v.to(device) for k, v in batch.items()}

        ############################################################

        finetuned_outputs = fully_finetuned_model(**new_batch)

        finetuned_start_logits = finetuned_outputs['start_logits']
        finetuned_end_logits = finetuned_outputs['end_logits']

        finetuned_start_predictions = torch.argmax(finetuned_start_logits, dim=-1)
        finetuned_end_predictions = torch.argmax(finetuned_end_logits, dim=-1)

        ############################################################

        semifrozen_outputs = semifrozen_model(**new_batch)

        semifrozen_start_logits = semifrozen_outputs['start_logits']
        semifrozen_end_logits = semifrozen_outputs['end_logits']

        semifrozen_start_predictions = torch.argmax(semifrozen_start_logits, dim=-1)
        semifrozen_end_predictions = torch.argmax(semifrozen_end_logits, dim=-1)

        ############################################################

        for i in range(0, len(finetuned_outputs['start_logits'])):

        	if finetuned_start_predictions[i] == new_batch['start_positions'][i] and finetuned_end_predictions[i] == new_batch['end_positions'][i]:
        		finetuned_correct_answer_context_lengths.append(new_batch['answer_lengths'][i])
        	else:
        		finetuned_wrong_answer_context_lengths.append(new_batch['answer_lengths'][i])

        	if semifrozen_start_predictions[i] == new_batch['start_positions'][i] and semifrozen_end_predictions[i] == new_batch['end_positions'][i]:
        		semifrozen_correct_answer_context_lengths.append(new_batch['answer_lengths'][i])
        	else:
        		semifrozen_wrong_answer_context_lengths.append(new_batch['answer_lengths'][i])


        #print("new_batch")
        #print(new_batch['answer_lengths'])

        progress_bar.update(1)

        print("---------------------------------------------------")
        print("Statistics for finetuned and semifrozen context lengths")
        print("---------------------------------------------------")
        print("Finetuned")
        print(len(finetuned_correct_answer_context_lengths))
        print(len(finetuned_wrong_answer_context_lengths))
        print(statistics.mean(finetuned_correct_answer_context_lengths.numpy()))
        print(statistics.stdev(finetuned_correct_answer_context_lengths.numpy()))
        print(statistics.mean(finetuned_wrong_answer_context_lengths.numpy()))
        print(statistics.stdev(finetuned_wrong_answer_context_lengths.numpy()))
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("Semifrozen")
        print(len(semifrozen_correct_answer_context_lengths))
        print(len(semifrozen_wrong_answer_context_lengths))
        print(statistics.mean(semifrozen_correct_answer_context_lengths.numpy()))
        print(statistics.stdev(semifrozen_correct_answer_context_lengths.numpy()))
        print(statistics.mean(semifrozen_wrong_answer_context_lengths.numpy()))
        print(statistics.stdev(semifrozen_wrong_answer_context_lengths.numpy()))
        print("---------------------------------------------------")


