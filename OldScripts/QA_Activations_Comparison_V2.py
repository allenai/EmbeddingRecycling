


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

from scipy import stats

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

          self.classifier = nn.Sequential(
                                            nn.Linear(in_features=embedding_size, out_features=2, bias=True)
                                         )



          

    def forward(self, input_ids, attention_mask, start_positions, end_positions, decoded_inputs=None, token_type_ids=None, question_lengths=None):

        output = self.encoderModel(input_ids, attention_mask)#['last_hidden_state']
        last_hidden_state = output['hidden_states'][len(output['hidden_states']) - 1]

        ##################################################################

        classifier_output = self.classifier(last_hidden_state)
        start_logits = classifier_output[:, :, 0]
        end_logits = classifier_output[:, :, 1]

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

num_epochs = 2 #1000 #10
patience_value = 5 #10 #3
current_dropout = True
number_of_runs = 1 #1 #5
frozen_choice = False
#chosen_learning_rate = 5e-6 #5e-6, 1e-5, 2e-5, 5e-5, 0.001
frozen_layers = 0 #12 layers for BERT total, 24 layers for T5 and RoBERTa
frozen_embeddings = False
average_hidden_state = False

assigned_batch_size = 8
gradient_accumulation_multiplier = 4

validation_set_scoring = False

warmup_steps_count_ratio = 0.2

############################################################

checkpoint_path = "checkpoints/testing_checkpoint.pt"

#model_choice = 'roberta-large'
model_choice = "bert-base-uncased"
#model_choice = "distilbert-base-uncased"
#model_choice = "nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large"

############################################################

#chosen_dataset = 'squad'
chosen_dataset = "trivia_qa"
context_cutoff_count = 1024
context_token_count = 512
multi_answer = False
remove_missing_answers = False

reduced_sample = True

dataset_version = "./" + chosen_dataset + "_dataset_" + model_choice + "_" + str(context_cutoff_count) + "_" + str(context_token_count)
dataset_version += "_" + str(multi_answer) + "_" + str(remove_missing_answers) + "_" + str(reduced_sample)

triviaqa_dataset = load_from_disk(dataset_version)
triviaqa_dataset.set_format("torch")

train_dataloader = DataLoader(triviaqa_dataset['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(triviaqa_dataset['validation'], batch_size=assigned_batch_size)

############################################################

chosen_learning_rate = 5e-5

fully_finetuned_model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
                                        average_hidden_state, frozen_embeddings)

#fully_finetuned_checkpoint_path = "paper_results_qa/bert-base-uncased/5e-06_0_False_10False_Run_0.pt"
#fully_finetuned_checkpoint_path = "fully_finetuned_roberta_checkpoint"
#fully_finetuned_checkpoint_path = "paper_results_qa/distilbert-base-uncased/5e-05_0_False_10False_Run_0.pt"
#fully_finetuned_checkpoint_path = "paper_results_qa/roberta-large/5e-06_0_False_10False_Run_3.pt"
#fully_finetuned_checkpoint_path = "paper_results_qa/bert-base-uncased/5e-06_0_False_10False_Run_7.pt"

#fully_finetuned_model.load_state_dict(torch.load(fully_finetuned_checkpoint_path))

fully_finetuned_model.to(device)

############################################################

frozen_layers = 6
frozen_embeddings = True

semifrozen_model = CustomBERTModel(model_choice, current_dropout, frozen_choice, frozen_layers, 
                                   average_hidden_state, frozen_embeddings)

#semifrozen_checkpoint_path = "paper_results_qa/bert-base-uncased/2e-05_6_True_10False_Run_9.pt"
#semifrozen_checkpoint_path = "semifrozen_roberta_checkpoint"
#semifrozen_checkpoint_path = "paper_results_qa/distilbert-base-uncased/5e-05_3_True_10False_Run_0.pt"
#semifrozen_checkpoint_path = "paper_results_qa/roberta-large/2e-05_18_True_10False_Run_0.pt"
#semifrozen_checkpoint_path = "paper_results_qa/bert-base-uncased/2e-05_6_True_10False_Run_7.pt"

#semifrozen_model.load_state_dict(torch.load(semifrozen_checkpoint_path))

semifrozen_model.to(device)

#print("Model checkpoints")
#print(fully_finetuned_checkpoint_path)
#print(semifrozen_checkpoint_path)

############################################################










criterion = nn.CrossEntropyLoss()
optimizer = Adam(fully_finetuned_model.parameters(), lr=chosen_learning_rate) #5e-6

num_training_steps = num_epochs * len(train_dataloader)

print("Warmup step count")
print(int(warmup_steps_count_ratio * len(train_dataloader)))

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=int(warmup_steps_count_ratio * len(train_dataloader)), num_training_steps=num_training_steps
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

    fully_finetuned_model.train()
    for batch in train_dataloader:

        #with torch.no_grad():

            new_batch = {k: v.to(device) for k, v in batch.items()}

            outputs = fully_finetuned_model(**new_batch)

            start_loss = criterion(outputs['start_logits'], batch['start_positions'].to(device))
            end_loss = criterion(outputs['end_logits'], batch['end_positions'].to(device))
            loss = (start_loss + end_loss) / 2.0

            loss.backward()

            gradient_accumulation_count += 1
            if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            train_losses.append(loss.item())


    progress_bar = tqdm(range(len(validation_dataloader)))

    fully_finetuned_model.eval()
    for batch in validation_dataloader:

        #with torch.no_grad():
            
            new_batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = fully_finetuned_model(**new_batch)

            start_loss = criterion(outputs['start_logits'], batch['start_positions'].to(device))
            end_loss = criterion(outputs['end_logits'], batch['end_positions'].to(device))
            loss = (start_loss + end_loss) / 2.0

            #loss = outputs.loss
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
    early_stopping(valid_loss, fully_finetuned_model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break



############################################################











criterion = nn.CrossEntropyLoss()
optimizer = Adam(semifrozen_model.parameters(), lr=chosen_learning_rate) #5e-6

num_training_steps = num_epochs * len(train_dataloader)

print("Warmup step count")
print(int(warmup_steps_count_ratio * len(train_dataloader)))

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=int(warmup_steps_count_ratio * len(train_dataloader)), num_training_steps=num_training_steps
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

    semifrozen_model.train()
    for batch in train_dataloader:

        #with torch.no_grad():

            new_batch = {k: v.to(device) for k, v in batch.items()}

            outputs = semifrozen_model(**new_batch)

            start_loss = criterion(outputs['start_logits'], batch['start_positions'].to(device))
            end_loss = criterion(outputs['end_logits'], batch['end_positions'].to(device))
            loss = (start_loss + end_loss) / 2.0

            loss.backward()

            gradient_accumulation_count += 1
            if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            train_losses.append(loss.item())


    progress_bar = tqdm(range(len(validation_dataloader)))

    semifrozen_model.eval()
    for batch in validation_dataloader:

        #with torch.no_grad():
            
            new_batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = semifrozen_model(**new_batch)

            start_loss = criterion(outputs['start_logits'], batch['start_positions'].to(device))
            end_loss = criterion(outputs['end_logits'], batch['end_positions'].to(device))
            loss = (start_loss + end_loss) / 2.0

            #loss = outputs.loss
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
    early_stopping(valid_loss, semifrozen_model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break



############################################################






















print("Beginning Evaluation")

metric = load_metric("accuracy")

total_start_position_predictions = torch.LongTensor([]).to(device)
total_start_position_references = torch.LongTensor([]).to(device)

total_end_position_predictions = torch.LongTensor([]).to(device)
total_end_position_references = torch.LongTensor([]).to(device)

inference_start = time.time()


finetuned_attentions_sums = []
semifrozen_attentions_sums = []





count = 0
count_max = 1010
top_layers_to_sum_over = 3

print("Max Count: " + str(count_max))
print("Summing over top layers: " + str(top_layers_to_sum_over))

if model_choice == 'roberta-large':
    print("Attention Heads: 16")
    attention_head_max = 16
elif model_choice == 'bert-base-uncased' or model_choice == "distilbert-base-uncased":
    print("Attention Heads: 12")
    attention_head_max = 12






progress_bar = tqdm(range(len(eval_dataloader)))
for batch in zip(eval_dataloader):

    with torch.no_grad():

        try:
            question_lengths = torch.reshape(batch[0]['question_lengths'], (1, 8))
        except:
            question_lengths = torch.reshape(batch[0]['question_lengths'], (1, 4))

        #print("Question lengths")
        #print(question_lengths)

        new_batch = {k: v.to(device) for k, v in batch[0].items()}

        ##################################################################

        count += 1

        if count > 1000 and count < count_max:

            count += 1

            finetuned_outputs = fully_finetuned_model(**new_batch)

            print("Attentions")
            print(len(finetuned_outputs['attentions']))
            print(type(finetuned_outputs['attentions']))
            print(len(finetuned_outputs['attentions'][0]))
            print(type(finetuned_outputs['attentions'][0]))
            print(len(finetuned_outputs['attentions'][0][5]))
            print(type(finetuned_outputs['attentions'][0][5]))
            print(len(finetuned_outputs['attentions'][0][5][0]))
            print(type(finetuned_outputs['attentions'][0][5][0]))
            print(len(finetuned_outputs['attentions'][0][5][0][0]))
            print(type(finetuned_outputs['attentions'][0][5][0][0]))

            for question_number in range(0, len(question_lengths[0])): # question number in the batch of 8
                attention_sum = 0
                for layer_index in range(0, top_layers_to_sum_over): # top 3 layers of 12 layers
                    for attention_head in range(0, attention_head_max): # cover all 12-16 attention heads
                        for i in range(0, question_lengths[0][question_number]): # Traverse token from 0 to question length
                            context_starting_position = question_lengths[0][question_number] + 1
                            attention_sum += sum(finetuned_outputs['attentions'][layer_index][question_number][attention_head][i][context_starting_position:])#.cpu().numpy()
                        for i in range(question_lengths[0][question_number], 512): # Traverse token in context
                            context_starting_position = question_lengths[0][question_number] + 1
                            attention_sum += sum(finetuned_outputs['attentions'][layer_index][question_number][attention_head][i][:context_starting_position])#.cpu().numpy()
                finetuned_attentions_sums.append(attention_sum)

            #finetuned_attentions_sums.append(attention_sum)
            #finetuned_attentions_sums.append(sum(sum(sum(finetuned_outputs['hidden_states'][11]))).cpu().numpy())

            ############################################################

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

            for question_number in range(0, len(question_lengths[0])): # question number in the batch of 8
                attention_sum = 0
                for layer_index in range(0, top_layers_to_sum_over): # top 3 layers of 12 layers
                    for attention_head in range(0, attention_head_max): # cover all 12-16 attention heads
                        for i in range(0, question_lengths[0][question_number]): # Traverse token from 0 to question length
                            context_starting_position = question_lengths[0][question_number] + 1
                            attention_sum += sum(semifrozen_outputs['attentions'][layer_index][question_number][attention_head][i][context_starting_position:])#.cpu().numpy()
                        for i in range(question_lengths[0][question_number], 512): # Traverse token in context
                            context_starting_position = question_lengths[0][question_number] + 1
                            attention_sum += sum(semifrozen_outputs['attentions'][layer_index][question_number][attention_head][i][:context_starting_position])#.cpu().numpy()
                semifrozen_attentions_sums.append(attention_sum)

            #semifrozen_attentions_sums.append(attention_sum)
            #semifrozen_attentions_sums.append(sum(sum(sum(semifrozen_outputs['hidden_states'][11]))).cpu().numpy())

            ############################################################

        progress_bar.update(1)


finetuned_attentions_sums = torch.stack(finetuned_attentions_sums)
semifrozen_attentions_sums = torch.stack(semifrozen_attentions_sums)

print("---------------------------------------")
print("Statistics for each attention")
print("---------------------------------------")
print("Finetuned")
print(len(finetuned_attentions_sums))
print(finetuned_attentions_sums[0:5])
print(torch.mean(finetuned_attentions_sums))
print(torch.std(finetuned_attentions_sums))
print("---------------------------------------")
print("Semifrozen")
print(len(semifrozen_attentions_sums))
print(semifrozen_attentions_sums[0:5])
print(torch.mean(semifrozen_attentions_sums))
print(torch.std(semifrozen_attentions_sums))
print("---------------------------------------")
print("T Test")
print(stats.ttest_ind(finetuned_attentions_sums.cpu().numpy(), semifrozen_attentions_sums.cpu().numpy()))
print("---------------------------------------")


