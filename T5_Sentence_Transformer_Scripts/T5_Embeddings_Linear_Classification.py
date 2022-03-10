
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
#datasets = ['sci-cite']

model_choice = "t5-3b"

for dataset in datasets:

	print("Calculating Performance for " + dataset)

	# Load embeddings and labels

	with open('embeddings/' + dataset + '_' + model_choice + '_train_set_embeddings.npy', 'rb') as f:
	    train_set_embeddings = np.load(f)

	with open('embeddings/' + dataset + '_' + model_choice + '_dev_set_embeddings.npy', 'rb') as f:
	    dev_set_embeddings = np.load(f)

	with open('embeddings/' + dataset + '_' + model_choice + '_test_set_embeddings.npy', 'rb') as f:
	    test_set_embeddings = np.load(f)

	

	with open('labels/' + dataset + '_' + model_choice + '_train_set_label.json') as json_file:
	    train_set_label = json.load(json_file)

	with open('labels/' + dataset + '_' + model_choice + '_dev_set_label.json') as json_file:
	    dev_set_label = json.load(json_file)

	with open('labels/' + dataset + '_' + model_choice + '_test_set_label.json') as json_file:
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

	totalDataset = train_set_label + dev_set_label

	############################################################

	print("Starting Training")

	total_training_set_embeddings = np.concatenate((train_set_embeddings, dev_set_embeddings), axis=0)
	total_training_set_labels = train_set_label + dev_set_label

	linearModel.fit(total_training_set_embeddings, total_training_set_labels)

	############################################################

	print("Starting Prediction")

	score = linearModel.score(test_set_embeddings, test_set_label)

	print("Performance for " + dataset + " dataset: " + str(round(score * 100, 2)))
	print("Total Possible Labels: " + str(len(labels_list)))

	majority_baseline_labels = [0 for label in test_set_label]

	correctCount = 0
	for predicted_label, true_label in zip(majority_baseline_labels, test_set_label):
		if predicted_label == true_label:
			correctCount += 1

	print("Majority Baseline")
	print(round(correctCount * 100 / len(majority_baseline_labels), 2))








