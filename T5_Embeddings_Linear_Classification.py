
from sklearn import linear_model
import numpy as np
import json

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import f1_score

############################################################

#linearModel = linear_model.LinearRegression()
linearModel = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3, random_state=0))

datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']

for dataset in datasets:

	print("Calculating Performance for " + dataset)

	# Load embeddings and labels

	with open('embeddings/' + dataset + '_train_set_embeddings.npy', 'rb') as f:
	    train_set_embeddings = np.load(f)

	with open('embeddings/' + dataset + '_dev_set_embeddings.npy', 'rb') as f:
	    dev_set_embeddings = np.load(f)

	with open('embeddings/' + dataset + '_test_set_embeddings.npy', 'rb') as f:
	    test_set_embeddings = np.load(f)

	

	with open('labels/' + dataset + '_train_set_label.json') as json_file:
	    train_set_label = json.load(json_file)

	with open('labels/' + dataset + '_dev_set_label.json') as json_file:
	    dev_set_label = json.load(json_file)

	with open('labels/' + dataset + '_test_set_label.json') as json_file:
	    test_set_label = json.load(json_file)

	############################################################

	print("Processing Labels")

	labels_list = sorted(list(set(train_set_label)))

	label_to_value_dict = {}

	count = 0
	for label in labels_list:
		label_to_value_dict[label] = count
		count += 1

	train_set_label = [label_to_value_dict[original_label] for original_label in train_set_label]
	dev_set_label = [label_to_value_dict[original_label] for original_label in dev_set_label]
	test_set_label = [label_to_value_dict[original_label] for original_label in test_set_label]

	############################################################

	print("Starting Training")

	total_training_set_embeddings = np.concatenate((train_set_embeddings, dev_set_embeddings), axis=0)
	total_training_set_labels = train_set_label + dev_set_label

	print("Compare shape of embeddings and labels for training")
	print(total_training_set_embeddings.shape)
	print(len(total_training_set_labels))

	linearModel.fit(total_training_set_embeddings, total_training_set_labels)

	############################################################

	print("Starting Prediction")

	print("Compare shape of embeddings and labels for testing")
	print(test_set_embeddings.shape)
	print(len(test_set_label))

	score = linearModel.score(test_set_embeddings, test_set_label)

	print("Performance for " + dataset + " dataset")
	print(score)
	print("Total Possible Labels")
	print(len(labels_list))








