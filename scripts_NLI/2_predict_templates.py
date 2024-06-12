import argparse
import tqdm
import os
import pandas as pd
from transformers import pipeline
import numpy as np
from datasets import Dataset
import csv

np.int = np.int32
batchsize = 32
modelnames = [
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "sileod/deberta-v3-base-tasksource-nli",
]

# "sileod/deberta-v3-large-tasksource-nli",  # not in lasha | bad results
# "microsoft/deberta-large-mnli", # older better ???
# "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  # no effect ????
# "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",  # broken
# "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli" # broken

parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
args = parser.parse_args()
dataset = args.dataset
part = args.part

dir_path = str(os.path.dirname(os.path.realpath(__file__)))
template_input_path = f"{dir_path}/../Results/{dataset}/NLI/{part}.tsv"
dataset_NLI = open(template_input_path, "r")

csvreader = csv.DictReader(dataset_NLI, delimiter="\t")
list_dict = []

# to pipeline
for row in iter(csvreader):
    list_dict.append(dict(text=row["prem"], text_pair=row["hyp"]))

dataset_dict = Dataset.from_list(list_dict)

for model in modelnames:
    # test all models and make dirs
    model_str = model.replace(r"/", "_")
    os.makedirs(f"{dir_path}/../Results/{dataset}/NLI/{model_str}/", exist_ok=True)
    pipe = pipeline("text-classification", model=model, batch_size=batchsize, top_k=None)

    top1: list[str] = []
    top1_score: list[float] = []
    top2: list[str] = []
    top2_score: list[float] = []
    top3: list[str] = []
    top3_score: list[float] = []

    # TODO: test???? does this make it slower??? TQDM VS time.time???? | Batches work, so looks fine??
    for out in tqdm.tqdm(pipe(iter(dataset_dict)), total=len(dataset_dict)):
        top1.append(out[0]["label"])
        top1_score.append(out[0]["score"])
        top2.append(out[1]["label"])
        top2_score.append(out[1]["score"])
        top3.append(out[2]["label"])
        top3_score.append(out[2]["score"])

    pd_train = pd.read_csv(template_input_path, sep="\t")
    pd_train["top1"] = top1
    pd_train["top1_score"] = top1_score
    pd_train["top2"] = top2
    pd_train["top2_score"] = top2_score
    pd_train["top3"] = top3
    pd_train["top3_score"] = top3_score
    pd_train.to_csv(f"{dir_path}/../Results/{dataset}/NLI/{model_str}/{part}.tsv", sep="\t", index=None)
