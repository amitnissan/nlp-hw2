import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score

from preprocess.data_preprocess import get_tokenized_datasets
from utils.train_utils import args
from utils.utils import set_seed, OUT_PATH, competition_file_name
from main import runner


def main(already_trained=False):
    # Interface for reproducing/loading the model, evaluate on office's dev and create tagged test csv for competition
    set_seed()
    print("\n ##### reproducing model ##### \n")
    if already_trained:
        baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets = get_tokenized_datasets()
    else:
        baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets = runner()

    print("\n ##### loading model ##### \n")
    model = AutoModelForSequenceClassification.from_pretrained(OUT_PATH)
    model.eval()
    trainer = Trainer(model=model, args=args)

    print("\n ##### evaluating model on dev ##### \n")
    raw_pred, labels, metrics = trainer.predict(tokenized_office_datasets['dev'])
    y_pred = np.array(np.argmax(raw_pred, axis=1))
    print(f"\n ### accuracy on dev: {accuracy_score(labels, y_pred)} ### \n")

    print(f"\n ##### predicting for test and creating file {competition_file_name} #####")
    raw_pred, _, _ = trainer.predict(tokenized_office_datasets['test'])
    y_pred = np.array(np.argmax(raw_pred, axis=1), dtype=bool)
    y_pred = np.array([str(y).upper() for y in y_pred])
    pd.DataFrame({'review': np.array(office_datasets['test'].data.columns[0]), 'label': y_pred}).to_csv(
        competition_file_name)


if __name__ == '__main__':
    main()
