from transformers import AutoModelForSequenceClassification

from preprocess.data_preprocess import get_datasets_from_files
from utils.utils import baby_files_paths, office_files_paths

if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    baby_datasets = get_datasets_from_files(baby_files_paths)
    office_datasets = get_datasets_from_files(office_files_paths)
    print("hi")
