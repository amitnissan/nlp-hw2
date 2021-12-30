from transformers import TrainingArguments

from utils.utils import  OUT_PATH

args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=32,
                         per_device_eval_batch_size=64, save_strategy='no', metric_for_best_model='dev_f1',
                         greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                         num_train_epochs=20, report_to='none')