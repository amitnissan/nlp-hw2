import numpy as np
import pandas as pd

from main import runner

if __name__ == '__main__':
    baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets, trainer = runner()
    trainer.predict(tokenized_office_datasets['test'])
    raw_pred, _, _ = trainer.predict(tokenized_office_datasets['test'])
    y_pred = np.array(np.argmax(raw_pred, axis=1), dtype=bool)
    pd.DataFrame({'review': np.array(office_datasets['test'].data.columns[0]), 'label': y_pred}).to_csv(
        "comp_207108820.csv")
