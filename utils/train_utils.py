from transformers import TrainingArguments, IntervalStrategy

from utils.utils import OUT_PATH

args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=32,
                         per_device_eval_batch_size=64, save_strategy=IntervalStrategy.NO,
                         metric_for_best_model='eval_accuracy',
                         greater_is_better=True, evaluation_strategy=IntervalStrategy.EPOCH, do_train=True,
                         num_train_epochs=20, report_to='none')
