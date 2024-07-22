import tqdm
import os
import pandas as pd
import numpy as np
from datasets import Dataset
import csv
import tqdm
import torch
import torch.distributed as dist
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


np.int = np.int64
batchsize = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dir_path = str(os.path.dirname(os.path.realpath(__file__)))
dataset_NLI = open("datasets/ppdb_phrase/ppdb_scrape_disjoint.tsv", "r")

csvreader = csv.DictReader(dataset_NLI, delimiter="\t")
list_dict = []

# to pipeline
for row in iter(csvreader):
    list_dict.append(dict(head=row["w1"], tail=row["w2"]))

dataset_dict = Dataset.from_list(list_dict)

toknames = ["distilbert-base-uncased", "roberta-base"]
modelnames = ["distilbert-base-uncased", "roberta-base"]

def preprocess_function(d, tokenizer):
    tokenized_batch = tokenizer(d['head'], d['tail'], padding=True, truncation=True, max_length=128)
    # tokenized_batch["label"] = label_encoder.encode(d['label'])
    return tokenized_batch

for tok, model_name in zip(toknames, modelnames):
    # test all models and make dirs
    os.makedirs(f"Results/paper/Baseline/{model_name}/", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tok)
    model = AutoModelForSequenceClassification.from_pretrained(f"models/paper/baseline/{model_name}/best", local_files_only=True).to(device)

    dataset_token = dataset_dict.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer},)
    dataset_token = dataset_token.remove_columns(["head", "tail"])
    dataset_token = dataset_token.with_format("torch")
    dataloader = DataLoader(dataset_token, batch_size=batchsize)

    preds = []
    with torch.no_grad():
        # print(dataloader)
        for i in tqdm.tqdm(dataloader):
            i_ids = i["input_ids"].to(device)
            a_mask = i["attention_mask"].to(device)
            if "token_type_ids" in i:
                t_ids = i["token_type_ids"].to(device)
                preds += model(i_ids, a_mask, t_ids)["logits"].tolist()
            else:
                preds += model(i_ids, a_mask)["logits"].tolist()
    result = pd.DataFrame(preds)
    result.to_csv(f"Results/paper/Baseline/{model_name}/predictions_raw.tsv", sep="\t")


    pred = np.argmax(result, axis=1)
    id2label: dict[int, str] = model.config.id2label
    pred_rel_test = [id2label[k] for k in pred]
    
    orgdata = pd.read_csv(f"datasets/ppdb_phrase/ppdb_scrape_disjoint.tsv", sep="\t")
    orgdata["pred"] = pd.Series(pred_rel_test)
    orgdata.to_csv(f"Results/paper/Baseline/{model_name}/predicts_all.tsv", sep="\t", index=False)

    # -------------------------------
    os.makedirs(f"Results/paper/Baseline/{model_name}/all", exist_ok=True)
    os.makedirs(f"Results/paper/Baseline/{model_name}/org", exist_ok=True)
    os.makedirs(f"Results/paper/Baseline/{model_name}/add", exist_ok=True)

    cr = classification_report(orgdata["label"], orgdata["pred"])
    cr_file = open(f"Results/paper/Baseline/{model_name}/all/classification_all.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(orgdata["label"], orgdata["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/Baseline/{model_name}/all/ConfusionMatrix_all.jpg")

    # -------------------------------
    org_df = orgdata[orgdata["meta"] == "org"]
    cr = classification_report(org_df["label"], org_df["pred"])
    cr_file = open(f"Results/paper/Baseline/{model_name}/org/classification_org.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(org_df["label"], org_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/Baseline/{model_name}/org/ConfusionMatrix_org.jpg")

    # -------------------------------
    add_df = orgdata[orgdata["meta"] == "add"]
    cr = classification_report(add_df["label"], add_df["pred"])
    cr_file = open(f"Results/paper/Baseline/{model_name}/add/classification_add.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(add_df["label"], add_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/paper/Baseline/{model_name}/add/ConfusionMatrix_add.jpg")