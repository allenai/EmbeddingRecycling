from tqdm import tqdm

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

                if len(current_words) >= 512:
                    print("Length error! Sequence truncated")
                    current_words = current_words[:512]
                    current_labels = current_labels[:512]

                total_words.append(current_words)
                total_labels.append(current_labels)

                current_words = []
                current_labels = []

            elif len(line_split) > 2:

                current_words.append(line_split[0])
                current_labels.append(line_split[3].replace("\n", ""))

    return total_words, total_labels

#############################################################

classification_datasets = ['bc5cdr', 'JNLPBA', 'NCBI-disease']

print("Testing split")
print("\n".split("\t"))

testing_words, testing_labels = process_NER_dataset('ner/' + classification_datasets[2] + '/test.txt')

print(testing_words[1])
print(testing_labels[1])



