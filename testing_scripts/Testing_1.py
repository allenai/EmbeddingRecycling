from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, load_metric
from sklearn.metrics import f1_score, accuracy_score
import statistics
import torch

##################################################

def compute_f1(predictions_list, references_list):

	f1_scores = []

	for prediction, truth in zip(predictions_list, references_list):

	    pred_tokens = [x for x in range(prediction[0], prediction[1] + 1)]
	    truth_tokens = [x for x in range(truth[0], truth[1] + 1)]

	    common_tokens = set(pred_tokens) & set(truth_tokens)
	    
	    # if there are no common tokens then f1 = 0
	    if len(common_tokens) == 0:
	        return 0
	    
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


y_true = [[0, 0], [1, 1], [0, 1]]
y_pred = [[0, 0], [1, 1], [1, 1]]

f1_score = f1_score(y_true, y_pred, average=None)
accuracy = accuracy_score(y_true, y_pred)

print(f1_score)
print(accuracy)

y_pred = torch.LongTensor([y_pred]).tolist()[0]
y_true = torch.LongTensor([y_true]).tolist()[0]

manual_f1_scores = compute_f1(y_true, y_pred)

print(manual_f1_scores)
print(exact_match(y_true, y_pred))


