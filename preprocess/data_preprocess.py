from typing import Dict
from datasets import Dataset
import pandas as pd


def get_datasets_from_files(files_path: Dict[str, str]) -> Dict[str, Dataset]:
    datasets_dict = {}
    for file in files_path:
        df = pd.read_csv(files_path[file], index_col=False)
        df = df.drop(columns=df.columns[0], axis=1)
        datasets_dict[file] = Dataset.from_pandas(df)
    return datasets_dict
