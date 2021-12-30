from typing import Dict
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

from utils.utils import baby_files_paths, office_files_paths, model_name


def get_datasets_from_files(files_path: Dict[str, str]) -> Dict[str, Dataset]:
    datasets_dict = {}
    for file in files_path:
        df = pd.read_csv(files_path[file], index_col=False)
        df = df.drop(columns=df.columns[0], axis=1)
        datasets_dict[file] = Dataset.from_pandas(df)
    return datasets_dict


def tokenize_data(datasets_dict: Dict[str, Dataset]):
    tokenized_datasets_dict = {}
    for dataset in datasets_dict:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_dataset = datasets_dict[dataset].map(tokenizer, input_columns='review',
                                                       fn_kwargs={"max_length": 128, "truncation": True,
                                                                  "padding": "max_length"})
        tokenized_dataset.set_format('torch')
        tokenized_datasets_dict[dataset] = tokenized_dataset
    return tokenized_datasets_dict


def re_adding_label_column(tokenized_datasets, datasets):
    for dataset_type in tokenized_datasets:
        if dataset_type not in ('unlabeled', 'test'):
            tokenized_datasets[dataset_type] = tokenized_datasets[dataset_type].add_column('label',
                                                                                           datasets[dataset_type][
                                                                                               'label'])


def get_tokenized_datasets():
    baby_datasets = get_datasets_from_files(baby_files_paths)
    office_datasets = get_datasets_from_files(office_files_paths)
    tokenized_baby_datasets = tokenize_data(baby_datasets)
    tokenized_office_datasets = tokenize_data(office_datasets)
    re_adding_label_column(tokenized_baby_datasets, baby_datasets)
    re_adding_label_column(tokenized_office_datasets, office_datasets)
    return baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets
