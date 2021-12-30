from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score

from preprocess.data_preprocess import get_datasets_from_files, tokenize_data
from utils.utils import baby_files_paths, office_files_paths, model_name, OUT_PATH


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'f1': f1_score(preds, labels, average='binary')}


if __name__ == '__main__':
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_seq_classification.to("cuda:0")
    baby_datasets = get_datasets_from_files(baby_files_paths)
    office_datasets = get_datasets_from_files(office_files_paths)
    tokenized_baby_datasets = tokenize_data(model_name, baby_datasets)
    tokenized_office_datasets = tokenize_data(model_name, office_datasets)
    for dataset_type in tokenized_baby_datasets:
        if dataset_type not in ('unlabeled', 'test'):
            tokenized_baby_datasets[dataset_type] = tokenized_baby_datasets[dataset_type].add_column('label',
                                                                                                     baby_datasets[
                                                                                                         dataset_type][
                                                                                                         'label'])

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=64,
                             per_device_eval_batch_size=128, save_strategy='no', metric_for_best_model='dev_f1',
                             greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=1, report_to='none')

    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=tokenized_baby_datasets['train'],
        eval_dataset=tokenized_baby_datasets['dev'],
        compute_metrics=metric_fn
    )

    print("starting train")
    trainer.train()

    print("hi")
