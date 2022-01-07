from transformers import AutoModelForSequenceClassification

from models.classification_model import train
from preprocess.data_preprocess import get_tokenized_datasets
from utils.utils import set_seed, model_name, OUT_PATH


def runner():
    # This function loads a pretrained model (see which model in utils.utils.model_name), fine train it for the
    # classification task, and saves the epoch's model with the best accuracy

    # load pretrain model
    set_seed()
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_seq_classification.to("cuda:0")

    # tokenize data sets
    baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets = get_tokenized_datasets()

    # train and save best model
    train(model_seq_classification, tokenized_baby_datasets)

    return baby_datasets, office_datasets, tokenized_baby_datasets, tokenized_office_datasets


if __name__ == '__main__':
    runner()
