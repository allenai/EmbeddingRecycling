
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

	#print(first_output['hidden_states'][i].shape)

	layer_difference = 0
	for j in range(0, len(first_output['hidden_states'][i])):
		layer_difference +=  1 - torch.nn.functional.cosine_similarity(first_output['hidden_states'][i][j], 
													         	  	   second_output['hidden_states'][i][j])

	#layer_difference = euclidean_distances(first_output['hidden_states'][i][0], 
	#									   second_output['hidden_states'][i][0])
	#new_layer_difference = []
	#for i in range(0, len(layer_difference)):
	#	new_layer_difference.append(layer_difference[i][i])
	#layer_difference = new_layer_difference

	total_difference_for_each_layer.append(layer_difference)


print("Total difference")
print(len(total_difference_for_each_layer))
#print(total_difference_for_each_layer)

last_difference = sum(total_difference_for_each_layer[0])
for i in range(0, len(total_difference_for_each_layer)):

	difference = total_difference_for_each_layer[i]

	print("Layer " + str(i))
	print(sum(difference))
	comparison_to_last_difference = (sum(difference) - last_difference) / last_difference
	print(str(comparison_to_last_difference))
	last_difference = sum(difference)
	print("-----------------------------")


#####################################################################

import matplotlib.pyplot as plt
   
Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
  
plt.plot(Year, Unemployment_Rate, color='red', marker='o')
plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
plt.show()
