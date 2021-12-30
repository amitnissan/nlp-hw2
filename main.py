from transformers import AutoModelForSequenceClassification

from preprocess.data_preprocess import get_datasets_from_files, tokenize_data
from utils.utils import baby_files_paths, office_files_paths, model_name

if __name__ == '__main__':
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_seq_classification.to("cuda:0")
    baby_datasets = get_datasets_from_files(baby_files_paths)
    office_datasets = get_datasets_from_files(office_files_paths)
    tokenized_baby_datasets = tokenize_data(model_name, baby_datasets)
    tokenized_office_datasets = tokenize_data(model_name, office_datasets)
    for dataset_type in tokenized_baby_datasets:
        if dataset_type not in ('unlabeled', 'test'):
            tokenized_baby_datasets[dataset_type] = tokenized_baby_datasets[dataset_type].add_column('label', baby_datasets[dataset_type]['label'])

    print("hi")
