# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/SICK/NLI/train.tsv --output Results/SICK/task_source/
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/paper/NLI_manual/templates_insert_test.tsv --output models/NLI/tasksource_full/
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/SICK_baseline/NLI/trial.tsv --output Results/SICK_baseline/task_source/ --preds True

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
batchsize = 64
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--output", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--preds", required=False, default=False, metavar="FILES", help="Only give predictions")
args = parser.parse_args()
dataset = args.dataset
output = args.output
preds_bool = args.preds
part = dataset.split(r"/")[-1].split(r".")[0]

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
# model_name = "sileod/deberta-v3-base-tasksource-nli"
model_name = "sileod/deberta-v3-base-tasksource-nli"
# model_name = "tasksource/deberta-small-long-nli"
model_str = model_name.replace(r"/", "_")

os.makedirs(f"{output}/full", exist_ok=True)
os.makedirs(f"{output}/inter", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model= torch.nn.DataParallel(model)
model.to(device)


dataset_token = dataset_dict.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer},)
dataset_token = dataset_token.remove_columns(["prem", "hyp"])
dataset_token = dataset_token.with_format("torch")
dataloader = DataLoader(dataset_token, batch_size=batchsize)


model_dict =  {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

preds = []
with torch.no_grad():
    for counter, i in enumerate(tqdm.tqdm(dataloader)):
        # print(counter*batchsize)
        i_ids = i["input_ids"].to(device)
        a_mask = i["attention_mask"].to(device)

        if preds_bool:
            if "token_type_ids" in i:
                t_ids = i["token_type_ids"].to(device)
                out = model(i_ids, a_mask, t_ids)
            else:
                out = model(i_ids, a_mask)

            preds += [model_dict[x.item()] for x in out.logits.argmax(1)]
        else:
            if "token_type_ids" in i:
                t_ids = i["token_type_ids"].to(device)
                preds += model(i_ids, a_mask, t_ids)["logits"].tolist()
            else:
                preds += model(i_ids, a_mask)["logits"].tolist()

if preds_bool:
    pd_train = pd.read_csv(dataset, sep="\t")
    pd_train["preds"] = preds
    pd_train.to_csv(f"{output}/full/predicts_{part}.tsv", sep="\t", index=None)

else:
    result = pd.DataFrame(preds)
    result.to_csv(f"{output}/full/predicts_{part}.tsv", sep="\t")

    result = pd.read_csv(dataset, delimiter="\t")
    result["preds"] = preds

    try:
        res_group = result.groupby("CombID").agg({'preds': 'sum'})
        res_group["preds"].to_csv(f"{output}/inter/predictions_{part}.tsv", sep="\t")
    except KeyError:
        # res_group = result.groupby("ID").agg({'preds': 'sum', 'label': lambda x: list(x)[0]})
        res_group = result.groupby("ID").agg({'preds': 'sum'})
        res_group = res_group[["label", "preds"]]
        res_group[["preds", "label"]].to_csv(f"{output}/inter/predictions_{part}.tsv", sep="\t")

