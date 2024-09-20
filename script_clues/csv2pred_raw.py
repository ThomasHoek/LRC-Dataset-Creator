import pandas as pd
import glob
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
parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
args = parser.parse_args()
dataset = args.dataset
part = args.part

model_path = f"models/clues/roberta-bless/"

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


def get_results(dataframe_path: str, part: str):
    dataframe_inp = pd.read_csv(dataframe_path, sep="\t")
    # print(dataframe_inp)
    dataframe_inp["head"] = dataframe_inp["W1"]
    dataframe_inp["tail"] = dataframe_inp["W2"]

    data = Dataset.from_pandas(dataframe_inp[["head", "tail"]])

    dataframe_inp.drop(["head", "tail"], axis=1, inplace=True)
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

    dataframe_inp["pred"] = pd.Series(pred_rel_test)

    os.makedirs(f"Results/{dataset}/clues/", exist_ok=True)
    # os.makedirs(f"Results/{dataset}/clues/{part}/all", exist_ok=True)
    # os.makedirs(f"Results/{dataset}/clues/{part}/add", exist_ok=True)
    # os.makedirs(f"Results/{dataset}/clues/{part}/org", exist_ok=True)

    dataframe_inp.to_csv(f"Results/{dataset}/clues/predicts_{part}.tsv", sep="\t", index=False)


load_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
load_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
trainargs = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=8)
pretrain = Trainer(tokenizer=load_tokenizer, model=load_model, args=trainargs)

if part == "all":
    files_found = glob.glob(f"Results/{dataset}/*.tsv")
    for i in files_found:
        part_strip = i.split("/")[-1].replace("_ccg.tsv", "").replace(f"{dataset}_", "")
        get_results(i, part_strip)

else:    
    files_found = glob.glob(f"Results/{dataset}/{dataset}_{part}_ccg.tsv")
    assert len(files_found) == 1
    get_results(files_found[0], part)


