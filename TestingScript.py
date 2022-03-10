from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, GPT2Tokenizer
import torch

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("hivemind/gpt-j-6B-8bit")

current_input = ["This is just a tester string to see how it works." for i in range(10)]

tokenized_input = tokenizer(current_input, truncation=True)

#print('tokenized_input')
#print(tokenized_input)

testing_input = torch.IntTensor(tokenized_input.input_ids)



current_result = model(testing_input)

print(type(current_result))
print(current_result['last_hidden_state'].shape)
#print(current_result['past_key_values'].shape)
#print((current_result.__dict__))







