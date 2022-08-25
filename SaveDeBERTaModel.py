
import transformers

from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer

model_choice = "microsoft/deberta-xlarge"
tokenizer = AutoTokenizer.from_pretrained(model_choice)
model = AutoModel.from_pretrained(model_choice)

model.save_pretrained("deberta/model/")
tokenizer.save_pretrained("deberta/tokenizer/")