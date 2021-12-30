import random
import numpy as np
import torch
import os

# This files holds general utilities

# GLOVE_PATH = "glove-twitter-200"
baby_files_paths = {"train": "clean_data/baby/train.csv",
                    "dev": "clean_data/baby/dev.csv", "unlabeled": "clean_data/baby/unlabeled.csv"}
office_files_paths = {"unlabeled": "clean_data/office_products/unlabeled.csv",
                      "test": "clean_data/office_products/test.csv",
                      "dev": "clean_data/office_products/dev.csv"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model_name = 'bert-base-uncased'

OUT_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))