# python -m torch.distributed.launch --nproc_per_node 2 ~/src/main_debug.py
# torchrun --dataset data_dir/SICK/trial.tsv

import argparse
import csv
import tqdm
import torch
import os
import numpy as np
import pandas as pd
import torch.distributed as dist
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from torch.utils.data import DataLoader

np.int = np.int64
batchsize = 32
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
args = parser.parse_args()
dataset = args.dataset

dataset_NLI = open(dataset, "r")
csvreader = csv.DictReader(dataset_NLI, delimiter="\t")
list_dict = []

# to pipeline
for row in iter(csvreader):
    list_dict.append(dict(prem=row["prem"], hyp=row["hyp"]))
dataset_dict = Dataset.from_list(list_dict)


def preprocess_function(rows, tokenizer):
    return tokenizer(rows["prem"], rows["hyp"], truncation=True, padding="max_length", max_length=64, return_tensors='pt')


# test all models and make dirs
for model_name in modelnames:
    model_str = model_name.replace(r"/", "_")
    os.makedirs(f"Results/baseline_manual/{model_str}/", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.distributed.get_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    dataset_token = dataset_dict.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer},)
    print(dataset_token)
    # break

    dataset_token = dataset_token.remove_columns(["prem", "hyp"])
    dataset_token = dataset_token.with_format("torch")
    dataloader = DataLoader(dataset_token, batch_size=batchsize)

    preds = []
    with torch.no_grad():
        for i in tqdm.tqdm(dataloader):
            i_ids = i["input_ids"].to(device)
            a_mask = i["attention_mask"].to(device)
            if "token_type_ids" in i:
                t_ids = i["token_type_ids"].to(device)
                preds += model(i_ids, a_mask, t_ids)["logits"].tolist()
            else:
                preds += model(i_ids, a_mask)["logits"].tolist()
    result = pd.DataFrame(preds)
    result.to_csv(f"Results/baseline_manual/{model_str}/predictions.tsv", sep="\t")