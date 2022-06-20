

from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, models, losses
from torch import nn
import ast

from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

device = torch.device('cuda:0')

word_embedding_model = models.Transformer('sentence-transformers/sentence-t5-xl', max_seq_length=256)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cuda')

################################################################################################################

#specified_dataset = 'chemprot'
#specified_dataset = 'sci-cite'
specified_dataset = 'sciie-relation-extraction'

print("Processing Dataset: " + specified_dataset)

# Chemprot train, dev, and test
with open('text_classification/' + specified_dataset + '/train.txt') as f:

	train_set = f.readlines()
	train_set = [ast.literal_eval(line) for line in train_set]
	train_set_text = [line['text'] for line in train_set]
	train_set_label = [line['label'] for line in train_set]

with open('text_classification/' + specified_dataset + '/dev.txt') as f:
	    
	dev_set = f.readlines()
	dev_set = [ast.literal_eval(line) for line in dev_set]
	dev_set_text = [line['text'] for line in dev_set]
	dev_set_label = [line['label'] for line in dev_set]


################################################################################################################


labels_list = sorted(list(set(train_set_label)))

label_to_value_dict = {}

count = 0
for label in labels_list:
	label_to_value_dict[label] = count
	count += 1

total_training_set_text = train_set_text + dev_set_text
total_training_set_labels = train_set_label + dev_set_label

inputs_training = []

total_training_set_text = [[text, label] for text, label in zip(total_training_set_text, total_training_set_labels)]

for text, label in zip(total_training_set_text, total_training_set_labels):
	#inputs_training.append(InputExample(texts="test text here", label= float(1)))
	inputs_training.append(InputExample(texts=text, label= float(label_to_value_dict[label])))

################################################################################################################

train_dataloader = DataLoader(inputs_training, shuffle=True, batch_size=8)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, show_progress_bar=True)

modelPath = "FinetunedModels/" + specified_dataset
model.save(modelPath)

################################################################################################################




