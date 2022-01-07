import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer

from utils.train_utils import args
from utils.utils import set_seed, OUT_PATH
from main import runner

if __name__ == '__main__':
    set_seed()
    print("recreating model and saving")
    baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets = runner()

    print("loading model")
    model = AutoModelForSequenceClassification.from_pretrained(OUT_PATH)
    trainer = Trainer(model=model, args=args)

    print("predicting for test")
    raw_pred, _, _ = trainer.predict(tokenized_office_datasets['test'])
    y_pred = np.array(np.argmax(raw_pred, axis=1), dtype=bool)
    y_pred = np.array([str(y).upper() for y in y_pred])

    file_name = "comp_207108820.csv"
    print(f"creating file: {file_name}")
    pd.DataFrame({'review': np.array(office_datasets['test'].data.columns[0]), 'label': y_pred}).to_csv(
        file_name)
