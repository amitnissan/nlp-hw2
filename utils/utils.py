import random
import numpy as np
import torch

# This files holds general utilities

# GLOVE_PATH = "glove-twitter-200"
baby_files_paths = {"unlabeled": "clean_data/baby/unlabeled.csv", "train": "clean_data/baby/train.csv",
                    "dev": "clean_data/baby/dev.csv"}
office_files_paths = {"unlabeled": "clean_data/baby/unlabeled.csv", "test": "clean_data/baby/test.csv",
                      "dev": "clean_data/baby/dev.csv"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
