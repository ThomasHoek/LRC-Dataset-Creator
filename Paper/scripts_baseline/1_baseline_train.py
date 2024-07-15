import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer


BATCH_SIZE = 16
MODEL_DIR = "models"
metric = load_metric('glue', "mnli")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_dir = "datasets/baseline"
# model_name = "roberta-base"
model_name = "distilbert-base-uncased"

file_dict = {'train': f'{dataset_dir}/merged_LRC_words/train.tsv',
             'val': f'{dataset_dir}/merged_LRC_words/validation.tsv',
             'test': f'{dataset_dir}/merged_LRC_words/test.tsv'}
dataset = load_dataset('csv', delimiter="\t", data_files=file_dict, column_names=['head', 'tail', 'label', 'meta'],)
dataset = dataset.remove_columns('meta')

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

class LabelEncoder:
    def __init__(self):
        self.labels_to_int = {}

    def encode(self, labels):
        prev_label = 0
        encoded_labels = []

        for label in labels:
            if label not in self.labels_to_int:
                self.labels_to_int[label] = prev_label
                encoded = prev_label
                prev_label += 1
            else:
                encoded = self.labels_to_int[label]

            encoded_labels.append(encoded)

        return encoded_labels

label_encoder = LabelEncoder()

# https://huggingface.co/transformers/preprocessing.html
def preprocess_function(d):
    tokenized_batch = tokenizer(d['head'], d['tail'], padding=True, truncation=True, max_length=128)
    tokenized_batch["label"] = label_encoder.encode(d['label'])
    return tokenized_batch

# tokenize the data | Force calculation for dicts
encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)


label2id = label_encoder.labels_to_int
id2label = {v: k for k, v in label2id.items()}

# load a model and prepare it for 5-way classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, id2label=id2label, label2id=label2id)
model.to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

args = TrainingArguments(
    f"{MODEL_DIR}/{model_name}", # to save models
    # evaluation_strategy = "epoch", # 1 epoch for training takes too long for colab
    evaluation_strategy = "steps",
    eval_steps = 500, # evaluate and save after training on every next 500x16 examples
    save_steps=500, # saves model after every 500 steps. save_steps should be divisible on eval_steps
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10, # going throught the training data only once
    weight_decay=0.01,
    load_best_model_at_end=True, # after fine-tuning trainer.model will keep the best model
    metric_for_best_model="accuracy",
    optim="adamw_torch",
    save_total_limit=3)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    # You could use "test" here but it will be cheating then
    # to select the model checkpoint which gets highest score on test
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

trainer.train()

pred_result = trainer.predict(encoded_dataset["test"])
pred_result

test_all = dataset['test'].to_pandas()
arr_max = np.argmax(pred_result[0], axis=-1)

test_all['pred'] = pd.Series(np.vectorize(id2label.get)(arr_max))

print(classification_report(test_all["label"], test_all["pred"], digits=3))

trainer.save_model(f"{MODEL_DIR}/{model_name}/best")

test_all.to_csv(f"{MODEL_DIR}/{model_name}/baseline_self_train.csv")
