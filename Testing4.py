from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline, AutoModel

model = AutoModel.from_pretrained("SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"),
tokenizer = AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune", skip_special_tokens=True)

tokenized_code = '''with open ( CODE_STRING , CODE_STRING ) as in_file : buf = in_file . readlines ( )  with open ( CODE_STRING , CODE_STRING ) as out_file : for line in buf :          if line ==   " ; Include this text   " :              line = line +   " Include below  "          out_file . write ( line ) '''
tokenized_code = tokenizer(tokenized_code)

print('tokenized_code')
print(tokenized_code)

batch = {k: v for k, v in tokenized_code.items()}
batch = {'ids': batch['input_ids'], 'mask': batch['attention_mask']}

print(model(**batch))