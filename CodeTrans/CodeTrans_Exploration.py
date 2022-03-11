from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline

pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_base_program_synthese_transfer_learning_finetune"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_base_program_synthese_transfer_learning_finetune", 
    										skip_special_tokens=True)#,
    #device=0
)

tokenized_code = "you are given an array of numbers a and a number b , compute the difference of elements in a and b"
pipeline([tokenized_code])

print(pipeline)
print(dir(pipeline))