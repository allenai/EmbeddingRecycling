
from sentence_transformers import SentenceTransformer, util

import ast
import torch
from tqdm import tqdm
import numpy as np
import json

model = SentenceTransformer('sentence-transformers/sentence-t5-xl', device='cuda')

############################################################

datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']

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

	train_set_embeddings = model.encode(train_set, show_progress_bar=True)
	dev_set_embeddings = model.encode(dev_set, show_progress_bar=True)
	test_set_embeddings = model.encode(test_set, show_progress_bar=True)

	print("Embeddings Shapes")
	print(train_set_embeddings.shape)
	print(dev_set_embeddings.shape)
	print(test_set_embeddings.shape)

	print("Completed chemprot_train_set_embeddings")

	with open('embeddings/' + dataset+ '_train_set_embeddings.npy', 'wb') as f:
	    np.save(f, train_set_embeddings)

	with open('embeddings/' + dataset + '_dev_set_embeddings.npy', 'wb') as f:
	    np.save(f, dev_set_embeddings)

	with open('embeddings/' + dataset + '_test_set_embeddings.npy', 'wb') as f:
	    np.save(f, test_set_embeddings)

	############################################################

	print("Save Labels")

	with open('labels/' + dataset + '_train_set_label.json', 'w') as f:
	    json.dump(train_set_label, f)

	with open('labels/' + dataset + '_dev_set_label.json', 'w') as f:
	    json.dump(dev_set_label, f)

	with open('labels/' + dataset + '_test_set_label.json', 'w') as f:
	    json.dump(test_set_label, f)

