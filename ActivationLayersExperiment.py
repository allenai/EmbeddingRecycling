
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
import torch
from sklearn.metrics.pairwise import euclidean_distances

tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)

inputs = tokenizer("Hello, my dog is so cute but he is very strange", return_tensors="pt")

with torch.no_grad():
    first_output = model(**inputs)

print("first output")
print(first_output['hidden_states'][0])
print(first_output['hidden_states'][0].shape)

###############################################################

inputs = tokenizer("Hello, my cat is not hairy and she is a goblin", return_tensors="pt")

with torch.no_grad():
    second_output = model(**inputs)

print("second output")
print(second_output['hidden_states'][0])
print(second_output['hidden_states'][0].shape)

###############################################################

total_difference_for_each_layer = []

for i in range(0, len(second_output['hidden_states'])):

	layer_difference = torch.nn.functional.cosine_similarity(first_output['hidden_states'][i][0], 
													         second_output['hidden_states'][i][0])

	#layer_difference = euclidean_distances(first_output['hidden_states'][i][0], 
	#									   second_output['hidden_states'][i][0])
	#new_layer_difference = []
	#for i in range(0, len(layer_difference)):
	#	new_layer_difference.append(layer_difference[i][i])
	#layer_difference = new_layer_difference

	total_difference_for_each_layer.append(layer_difference)


print("Total difference")
print(len(total_difference_for_each_layer))

last_score = len(total_difference_for_each_layer[0])

for i in range(0, len(total_difference_for_each_layer)):

	difference = total_difference_for_each_layer[i]

	print("Layer " + str(i))
	print(difference)
	print("Total Similarity Score: " + str(sum(difference) / len(difference)))
	score_change = ((sum(difference) / len(difference))  - last_score) / last_score
	print("Difference from last score: " + str(score_change))
	last_score = sum(difference) / len(difference)
	print("-------------------")



#print(model_encoding.__dict__)

#print("---------------------------")

#print("Activation Function")
#print(model_encoding.encoder.layer[0].intermediate.intermediate_act_fn)