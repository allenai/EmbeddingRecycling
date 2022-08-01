from collections import abc
import transformers
import torch


def add_fp16_simulation(model: torch.nn.Module, path: str):
    """Simulates casting to fp16 for storage"""
    submodule = model.get_submodule(path)

    prev_forward = submodule.forward
    def forward(*args, **kwargs):
        out = prev_forward(*args, **kwargs)

        if isinstance(out, torch.Tensor):
            out = out.to(torch.float16).to(torch.float32)
        if isinstance(out, tuple):
            out = [t.to(torch.float16).to(torch.float32) for t in out]
        elif isinstance(out, abc.Mapping):
            out = {k: v.to(torch.float16).to(torch.float32) for k, v in out.items()}
        return out

    submodule.forward = forward


model = transformers.BertForSequenceClassification.\
    from_pretrained("bert-base-uncased").eval()

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_input = tokenizer(['This is a test'], return_tensors='pt')

out_fp32 = model(**tokenized_input)

add_fp16_simulation(model, "bert.encoder.layer.6")
out_fp16 = model(**tokenized_input)

diff = (out_fp16.logits - out_fp32.logits).abs().mean().tolist()
print(f'diff: {diff:.2e}')
