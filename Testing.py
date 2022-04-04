
import ast

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction']

for dataset in classification_datasets:

    # Chemprot train, dev, and test
    with open('text_classification/' + dataset + '/train.txt') as f:

        train_set = f.readlines()
        train_set = [ast.literal_eval(line) for line in train_set]
        train_set_text = [line['text'] for line in train_set]
        train_set_label = [line['label'] for line in train_set]

    with open('text_classification/' + dataset + '/dev.txt') as f:
        
        dev_set = f.readlines()
        dev_set = [ast.literal_eval(line) for line in dev_set]
        dev_set_text = [line['text'] for line in dev_set]
        dev_set_label = [line['label'] for line in dev_set]

    with open('text_classification/' + dataset + '/test.txt') as f:
        
        test_set = f.readlines()
        test_set = [ast.literal_eval(line) for line in test_set]
        test_set_text = [line['text'] for line in test_set]
        test_set_label = [line['label'] for line in test_set]



    print(dataset)
    print(len(train_set))
    print(len(dev_set))
    print(len(test_set))

    import torch

    model_choice = 'roberta-large'

    preloaded_training_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_training.pt')
    preloaded_validation_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_validation.pt')
    preloaded_test_tensors = torch.load('Experiment2_Tensors/' + dataset + '_' + model_choice + '_testing.pt')

    print(preloaded_training_tensors.shape)
    print(preloaded_validation_tensors.shape)
    print(preloaded_test_tensors.shape)





