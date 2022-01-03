from transformers import Trainer
from sklearn.metrics import f1_score, accuracy_score

from utils.train_utils import args
from utils.utils import set_seed


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'accuracy': accuracy_score(preds, labels)}


def train(model_seq_classification, tokenized_baby_datasets):
    set_seed()
    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=tokenized_baby_datasets['train'],
        eval_dataset=tokenized_baby_datasets['dev'],
        compute_metrics=metric_fn
    )
    print("starting train")
    trainer.train()
    return trainer
