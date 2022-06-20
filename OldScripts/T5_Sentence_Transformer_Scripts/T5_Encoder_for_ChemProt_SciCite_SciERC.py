

from transformers import T5Tokenizer, T5EncoderModel
import torch
from tqdm import tqdm

import ast
import numpy as np
import json

device = "cuda:0"

datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']
model_choice = "t5-3b"

tokenizer = T5Tokenizer.from_pretrained(model_choice, model_max_length=512)
model = T5EncoderModel.from_pretrained(model_choice).to(device)

############################################################

def gatherHiddenStates(input_texts):

	embeddings_list = []

	input_ids = tokenizer(
				    input_texts, return_tensors="pt", padding="max_length", truncation=True
				).input_ids.to(device)

	for i in tqdm(range(0, len(input_texts))):

		if i % 32 == 0:

			with torch.no_grad():

				outputs = model(input_ids=input_ids[i:i + 32])

				last_hidden_states = outputs.last_hidden_state

				#if i <= 1000:

					#print("Current outputs")
					#print(outputs)
					#print("Current Hidden State")
					#print(last_hidden_states.shape)
					#print(last_hidden_states)

				averaged_hidden_state = torch.mean(last_hidden_states, 1)

				#print(averaged_hidden_state.shape)

				embeddings_list.append(averaged_hidden_state.to('cpu'))


	embeddings_array = torch.cat(embeddings_list, dim=0).numpy()

	print("embeddings_array")
	print(embeddings_array.shape)

	return embeddings_array


############################################################

for dataset in datasets:

	print("Processing " + dataset)

	# Chemprot train, dev, and test
	with open('text_classification/' + dataset + '/train.txt') as f:

	    train_set = f.readlines()
	    train_set = [ast.literal_eval(line) for line in train_set]
	    train_set_text = [line['text'] for line in train_set]
	    train_set_label = [line['label'] for line in train_set]

	with open('text_classification/' + dataset + '/dev.txt') as f:
	    
	    dev_set = f.readlines()
	    dev_set = [ast.literal_eval(line) for line in dev_set]
	    dev_set_text = [line['text'] for line in dev_set]
	    dev_set_label = [line['label'] for line in dev_set]

	with open('text_classification/' + dataset + '/test.txt') as f:
	    
	    test_set = f.readlines()
	    test_set = [ast.literal_eval(line) for line in test_set]
	    test_set_text = [line['text'] for line in test_set]
	    test_set_label = [line['label'] for line in test_set]



	print("Size of Subsets")
	print(len(train_set))
	print(len(dev_set))
	print(len(test_set))

	############################################################

	print("Encoding the Train, Dev, and Test Sets")

	train_set_embeddings = gatherHiddenStates(train_set_text)
	dev_set_embeddings = gatherHiddenStates(dev_set_text)
	test_set_embeddings = gatherHiddenStates(test_set_text)

	print("Embeddings Shapes")
	print(train_set_embeddings.shape)
	print(dev_set_embeddings.shape)
	print(test_set_embeddings.shape)

	print("Completed chemprot_train_set_embeddings")

	with open('embeddings/' + dataset + '_' + model_choice + '_train_set_embeddings.npy', 'wb') as f:
	    np.save(f, train_set_embeddings)

	with open('embeddings/' + dataset + '_' + model_choice + '_dev_set_embeddings.npy', 'wb') as f:
	    np.save(f, dev_set_embeddings)

	with open('embeddings/' + dataset + '_' + model_choice + '_test_set_embeddings.npy', 'wb') as f:
	    np.save(f, test_set_embeddings)

	############################################################

	print("Save Labels")

	with open('labels/' + dataset + '_' + model_choice + '_train_set_label.json', 'w') as f:
	    json.dump(train_set_label, f)

	with open('labels/' + dataset + '_' + model_choice + '_dev_set_label.json', 'w') as f:
	    json.dump(dev_set_label, f)

	with open('labels/' + dataset + '_' + model_choice + '_test_set_label.json', 'w') as f:
	    json.dump(test_set_label, f)




