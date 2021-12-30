from transformers import AutoModelForSequenceClassification

from models.classification_model import train
from preprocess.data_preprocess import get_tokenized_datasets
from utils.utils import model_name


def runner():
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_seq_classification.to("cuda:0")

    baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets = get_tokenized_datasets()

    trainer = train(model_seq_classification, tokenized_baby_datasets)

    return baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets, trainer


if __name__ == '__main__':
    runner()
