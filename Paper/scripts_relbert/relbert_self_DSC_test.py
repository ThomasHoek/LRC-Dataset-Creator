from math import e
from relbert import RelBERT
# import relbert.lm
import numpy as np
import pickle
from datasets import load_from_disk
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)
from datasets import Dataset
import os
import re
import numpy as np
import argparse


import  matplotlib.pyplot as plt

cur_dir = os.path.dirname(os.path.realpath(__file__))


def embedding(model: RelBERT, data: DataFrame, split: str, batch_size: int = 32):  # -> list[NDArray[Any]]:
    # data = dataset[split]
    x_tuple = [tuple(_x) for _x in zip(data["head"], data["tail"])]
    x_back = [tuple(_x) for _x in zip(data["tail"], data["head"])]

    x = model.get_embedding(x_tuple, batch_size=batch_size)
    x_back = model.get_embedding(x_back, batch_size=batch_size)

    x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
    return x


for model_name in ["relbert-roberta-base", "relbert-roberta-large"]:
    with open(fr"models/relbert/{model_name}/model.pkl", "rb") as input_file:
        mlp = pickle.load(input_file)

    with open(fr"models/relbert/{model_name}/model_dict.pkl", "rb") as input_file:
        lab2num = pickle.load(input_file)
        num2lab = {v: k for k, v in lab2num.items()}


    model = RelBERT(model=f"relbert/{model_name}")

    orgdata = pd.read_csv(f"datasets/ppdb_phrase/ppdb_scrape_disjoint.tsv", sep="\t")
    orgdata["head"] = orgdata["w1"]
    orgdata["tail"] = orgdata["w2"]


    data = Dataset.from_pandas(orgdata[["head", "tail"]])
    data_type = "test"

    x = embedding(model, data, data_type)
    pred = mlp.predict(x)
    pred_rel_test: list[str] = [num2lab[k] for k in pred]

    orgdata["pred"] = pd.Series(pred_rel_test)
    orgdata.drop(columns=["head", "tail"], inplace=True)

    os.makedirs(f"Results/relbert/{model_name}/all", exist_ok=True)
    os.makedirs(f"Results/relbert/{model_name}/add", exist_ok=True)
    os.makedirs(f"Results/relbert/{model_name}/org", exist_ok=True)

    orgdata.to_csv(f"Results/relbert/{model_name}/predicts_all.tsv", sep="\t", index=False)

    # -------------------------------
    cr = classification_report(orgdata["label"], orgdata["pred"])
    cr_file = open(f"Results/relbert/{model_name}/all/classification_all.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(orgdata["label"], orgdata["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/relbert/{model_name}/all/ConfusionMatrix_all.jpg")

    # -------------------------------
    org_df = orgdata[orgdata["meta"] == "org"]
    cr = classification_report(org_df["label"], org_df["pred"])
    cr_file = open(f"Results/relbert/{model_name}/org/classification_org.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(org_df["label"], org_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/relbert/{model_name}/org/ConfusionMatrix_org.jpg")

    # -------------------------------
    add_df = orgdata[orgdata["meta"] == "add"]
    cr = classification_report(add_df["label"], add_df["pred"])
    cr_file = open(f"Results/relbert/{model_name}/add/classification_add.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(add_df["label"], add_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"Results/relbert/{model_name}/add/ConfusionMatrix_add.jpg")