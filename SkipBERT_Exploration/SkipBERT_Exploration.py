
import psutil, os
import torch
import skipbert
#from skipbert import *
from skipbert.modeling import SkipBertModel
from transformers import BertTokenizerFast, BertConfig
import time
import ast
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

p = psutil.Process(os.getpid())
p.nice(100)  # set process priority
print('nice:', p.nice())
torch.set_num_threads(1) # set num of torch threads

############################################################

device = "cuda:0"
#device = "cpu"
device = torch.device(device)

#PATH_TO_MODEL = "skipbert-L6-6"
PATH_TO_MODEL = "skipbert-L6-2"

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
dataset = 'sciie-relation-extraction'
assigned_batch_size = 32

############################################################

# Input Related
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

inputs = tokenizer(
    ["Good temper decides everything"],
    return_tensors='pt', padding='max_length', max_length=128
)

inputs = {
   k: (v.to(device) if isinstance(v, torch.Tensor) and k != 'input_ids' else v) for k, v in inputs.items()
}

# Model Related
config = BertConfig.from_pretrained(PATH_TO_MODEL)
config.plot_mode = 'plot_passive'

model = SkipBertModel.from_pretrained(PATH_TO_MODEL, config=config).to(device)
model.eval()

# Inference
# first time will compute the shallow layers
# execution_start = time.time()
# ret = model(**inputs)
# first_time = time.time() - execution_start
# print("First Run: " + str(first_time))


# # second time will retrieve hidden states from PLOT
# execution_start = time.time()
# ret = model(**inputs)
# second_time = time.time() - execution_start
# print("Second Run: " + str(second_time))

# print("Speedup: " + str((first_time - second_time) / first_time))







#################################################################

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

#################################################################

# Chemprot train, dev, and test
with open('../EmbeddingRecycling/text_classification/' + dataset + '/train.txt') as f:

    train_set = f.readlines()
    train_set = [ast.literal_eval(line) for line in train_set]
    train_set_text = [line['text'] for line in train_set]
    train_set_label = [line['label'] for line in train_set]

with open('../EmbeddingRecycling/text_classification/' + dataset + '/dev.txt') as f:
    
    dev_set = f.readlines()
    dev_set = [ast.literal_eval(line) for line in dev_set]

    dev_set_text = []
    dev_set_label = []
    for line in dev_set:

        # Fix bug in MAG dev where there is a single label called "category"
        if line['label'] != 'category':
            dev_set_text.append(line['text'])
            dev_set_label.append(line['label'])
        else:
            print("Found the error with category")

with open('../EmbeddingRecycling/text_classification/' + dataset + '/test.txt') as f:
    
    test_set = f.readlines()
    test_set = [ast.literal_eval(line) for line in test_set]
    test_set_text = [line['text'] for line in test_set]
    test_set_label = [line['label'] for line in test_set]


############################################################

labels_list = sorted(list(set(train_set_label)))

label_to_value_dict = {}

count = 0
for label in labels_list:
  label_to_value_dict[label] = count
  count += 1

train_set_label = [label_to_value_dict[label] for label in train_set_label]
dev_set_label = [label_to_value_dict[label] for label in dev_set_label]
test_set_label = [label_to_value_dict[label] for label in test_set_label]

############################################################

training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:1000]
training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:1000]
validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

############################################################


classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                'validation': validation_dataset_arrow, 
                                'test' : test_dataset_arrow})
tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["text", "labels"])
tokenized_datasets.set_format("torch")

############################################################

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)








############################################################

execution_start = time.time()

for batch in tqdm(eval_dataloader):
	new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
	ret = model(**new_batch)

first_eval_time = time.time() - execution_start
print("First Evaluation Run: " + str(first_eval_time))

############################################################

execution_start = time.time()

for batch in tqdm(eval_dataloader):
	new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
	ret = model(**new_batch)

second_eval_time = time.time() - execution_start
print("Second Evaluation Run: " + str(second_eval_time))

print("Speedup: " + str(round(((first_eval_time - second_eval_time) / first_eval_time) * 100, 2)) + "%")

















