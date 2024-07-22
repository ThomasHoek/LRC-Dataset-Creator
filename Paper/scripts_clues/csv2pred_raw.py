import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from datasets import Dataset
import os
import re
import numpy as np
import argparse


cur_dir: str = os.path.dirname(os.path.realpath(__file__))
model_path = f"models/clues/roberta-bless/"
Shwartz: bool = "shwartz" in model_path or "Ppdb" in model_path


load_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
load_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
trainargs = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=8)
pretrain = Trainer(tokenizer=load_tokenizer, model=load_model, args=trainargs)


def preprocess_function(rows, tokenizer):
    """tokenize the column 'verb' of the rows"""
    inputs = tokenizer(rows["verb"], truncation=True, padding="max_length", max_length=64)
    return inputs


def verb_row(row, template, tokenizer):
    w1 = str(row["head"])
    w2 = str(row["tail"])

    sentence = re.sub("<W1>", w1, template)
    sentence = re.sub("<W2>", w2, sentence)
    sentence = re.sub("<SEP>", tokenizer.sep_token, sentence)

    return {"verb": sentence}

orgdata = pd.read_csv(f"datasets/ppdb_phrase/ppdb_scrape_disjoint.tsv", sep="\t")
orgdata["head"] = orgdata["w1"]
orgdata["tail"] = orgdata["w2"]

data = Dataset.from_pandas(orgdata[["head", "tail"]])

orgdata.drop(["head", "tail"], axis=1, inplace=True)
sentences_info = open(f"{model_path}/sents.txt")
train_template = sentences_info.readline().rstrip()

data = data.map(verb_row, fn_kwargs={"template": train_template, "tokenizer": load_tokenizer})
data = data.map(preprocess_function, batched=True, batch_size=64, fn_kwargs={"tokenizer": load_tokenizer})

encoded_verb_test = data
encoded_verb_test.set_format("torch")

predicciones = pretrain.predict(encoded_verb_test)
pred = np.argmax(predicciones.predictions, axis=1)
id2label: dict[int, str] = load_model.config.id2label
pred_rel_test = [id2label[k] for k in pred]

orgdata["pred"] = pd.Series(pred_rel_test)

os.makedirs("Results/paper/clues/all", exist_ok=True)
os.makedirs("Results/paper/clues/add", exist_ok=True)
os.makedirs("Results/paper/clues/org", exist_ok=True)

orgdata.to_csv(f"Results/paper/clues/predicts_all.tsv", sep="\t", index=False)

# -------------------------------
cr = classification_report(orgdata["label"], orgdata["pred"])
cr_file = open(f"Results/paper/clues/all/classification_all.txt", "w+")
cr_file.write(cr)
cr_file.close()

disp = ConfusionMatrixDisplay.from_predictions(orgdata["label"], orgdata["pred"])
disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/clues/all/ConfusionMatrix_all.jpg")

# -------------------------------
org_df = orgdata[orgdata["meta"] == "org"]
cr = classification_report(org_df["label"], org_df["pred"])
cr_file = open(f"Results/paper/clues/org/classification_org.txt", "w+")
cr_file.write(cr)
cr_file.close()

disp = ConfusionMatrixDisplay.from_predictions(org_df["label"], org_df["pred"])
disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/clues/org/ConfusionMatrix_org.jpg")

# -------------------------------
add_df = orgdata[orgdata["meta"] == "add"]
cr = classification_report(add_df["label"], add_df["pred"])
cr_file = open(f"Results/paper/clues/add/classification_add.txt", "w+")
cr_file.write(cr)
cr_file.close()

disp = ConfusionMatrixDisplay.from_predictions(add_df["label"], add_df["pred"])
disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/clues/add/ConfusionMatrix_add.jpg")