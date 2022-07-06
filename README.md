# Embedding Recycling
Reusing Sequence Representations for Large Language Models

## Datasets

To access our datasets using for text classification, please go to the folder titled `text_classification`. 

To access our datasets using for named-entity recognition (NER), please go to the folder titled `ner`. 

To access our datasets using for question answering (QA), please access the TriviaQA and SQuAD datasets on HuggingFace at the following links:

- [TriviaQA](https://huggingface.co/datasets/trivia_qa)
- [SQuAD](https://huggingface.co/datasets/squad)

## Setup Environment

Run the following commands to setup a conda environment:

````
conda create --name embedding_recycling --file requirements.txt
conda activate embedding_recycling
````

## Experiment Replication

To replicate our results, use the conda environment listed above and run the following scripts for each set of dataset group:

- For text classification, use the `TextClassificationScripts/GeneralLinearClassifier_PaperResults.py` script
- For NER, use the `NER_Scripts/General_NER_Classifier_PaperResults.py` script
- For QA, use the `QA_Scripts/GeneralQuestionAnswering_PaperResults.py` script

The hyperparameters for replicating each experiment are included in the `HyperparameterSelection` folder

## Citing
