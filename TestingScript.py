


from datasets import load_dataset,load_metric


dataset = load_dataset("yelp_review_full")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

print('dataset')
print(type(dataset))
print(type(dataset['train']))
print(type(dataset['train'][0]))




print("Dataset structure")
print(type(dataset))
print(dataset.shape)
print(type(dataset['train']))
print(dataset['train'][0])


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")




print('tokenized_datasets')
print(type(tokenized_datasets))
print(type(tokenized_datasets['train']))
print(type(tokenized_datasets['train'][0]))
#print((tokenized_datasets['train'][0]))


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))



print("small_train_dataset")
print(type(small_train_dataset))
print(small_train_dataset.shape)
print(small_train_dataset[0])


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)



from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)



import torch

device = torch.device("cuda")
model.to(device)



from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)



metric = load_metric("accuracy")
model.eval()

total_predictions = torch.FloatTensor([]).to(device)
total_references = torch.FloatTensor([]).to(device)

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

    total_predictions = torch.cat((total_predictions, predictions), 0)
    total_references = torch.cat((total_references, batch["labels"]), 0)



print("--------------------------")
print(total_predictions.shape)
print(total_references.shape)

results = metric.compute(references=total_predictions, predictions=total_references)
print("Results for Dev Set: " + str(results['accuracy']))

#metric.compute()

#print(metric)
#print(dir(metric))

############################################################






