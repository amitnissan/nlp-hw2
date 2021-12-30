from typing import Dict
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


def get_datasets_from_files(files_path: Dict[str, str]) -> Dict[str, Dataset]:
    datasets_dict = {}
    for file in files_path:
        df = pd.read_csv(files_path[file], index_col=False)
        df = df.drop(columns=df.columns[0], axis=1)
        datasets_dict[file] = Dataset.from_pandas(df)
    return datasets_dict


def tokenize_data(model_name: str, datasets_dict: Dict[str,Dataset]):
    tokenized_datasets_dict = {}
    for dataset in datasets_dict:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_dataset = datasets_dict[dataset].map(tokenizer, input_columns='review',
                                                       fn_kwargs={"max_length": 128, "truncation": True,
                                                                  "padding": "max_length"})
        tokenized_dataset.set_format('torch')
        # tokenized_dataset.to("cuda:0") FIXME throws an error
        tokenized_datasets_dict[dataset] = tokenized_dataset
    return tokenized_datasets_dict
