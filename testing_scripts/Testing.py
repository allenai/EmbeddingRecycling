
import ast
from tqdm import tqdm

classification_datasets = ['chemprot', 'sci-cite', 'sciie-relation-extraction', 'mag']

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

        dev_set_text = []
        dev_set_label = []
        for line in dev_set:

            # Fix bug in MAG dev where there is a single label called "category"
            if line['label'] != 'category':
                dev_set_text.append(line['text'])
                dev_set_label.append(line['label'])
            else:
                print("Found the error with category")

    with open('text_classification/' + dataset + '/test.txt') as f:
        
        test_set = f.readlines()
        test_set = [ast.literal_eval(line) for line in test_set]
        test_set_text = [line['text'] for line in test_set]
        test_set_label = [line['label'] for line in test_set]


    print("Train, dev, and test set sizes for " + dataset)
    print(len(train_set_text))
    print(len(dev_set_text))
    print(len(test_set_text))
    print("------------------------------------------")






####################################################################################################

def process_NER_dataset(dataset_path):

    total_words = []
    total_labels = []

    current_words = []
    current_labels = []

    with open(dataset_path) as f:

        train_set = f.readlines()

        for line in tqdm(train_set):

            line_split = line.split("\t")

            if len(line_split) <= 2 and len(current_words) != 0:

                if len(current_words) != len(current_labels):
                    print("Error")

                #if len(current_words) >= 512:
                #    print("Length error! Sequence truncated")
                #    current_words = current_words[:512]
                #    current_labels = current_labels[:512]

                total_words.append(current_words)
                total_labels.append(current_labels)

                current_words = []
                current_labels = []

            elif len(line_split) > 2:

                current_words.append(line_split[0])
                current_labels.append(line_split[3].replace("\n", ""))

    return total_words, total_labels

####################################################################################################

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']

for dataset in classification_datasets:

    # Gather train, dev, and test sets
    train_set_text, train_set_label = process_NER_dataset('ner/' + dataset + '/train.txt')

    dev_set_text, dev_set_label = process_NER_dataset('ner/' + dataset + '/dev.txt')

    test_set_text, test_set_label = process_NER_dataset('ner/' + dataset + '/test.txt')

    print("Train, dev, and test set sizes for " + dataset)
    print(len(train_set_text))
    print(len(dev_set_text))
    print(len(test_set_text))
    print("------------------------------------------")




