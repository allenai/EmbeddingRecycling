from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModel.from_pretrained("EleutherAI/gpt-j-6B")

current_input = ["This is just a tester string to see how it works." for i in range(10)]

tokenized_input = tokenizer(current_input, padding=True, truncation=True)

print('tokenized_input')
print(tokenized_input)

testing_input = torch.IntTensor(tokenized_input.input_ids)



current_result = model(testing_input)

print(type(current_result))
print(current_result['last_hidden_state'].shape)
#print(current_result['past_key_values'].shape)
print((current_result.__dict__))







