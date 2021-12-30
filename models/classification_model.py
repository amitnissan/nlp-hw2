from transformers import Trainer
from sklearn.metrics import f1_score

from utils.train_utils import args


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'f1': f1_score(preds, labels, average='binary')}


def train(model_seq_classification, tokenized_baby_datasets):
    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=tokenized_baby_datasets['train'],
        eval_dataset=tokenized_baby_datasets['dev'],
        compute_metrics=metric_fn
    )
    print("starting train")
    trainer.train()
