# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/SICK/NLI/train.tsv --output Results/SICK/task_source/
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/paper/NLI_manual/templates_insert_test.tsv --output models/NLI/tasksource_full/
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset Results/SICK_baseline/NLI/trial.tsv --output Results/SICK_baseline/task_source/ --preds True
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset lex_preds/SICK/NLI/templates/trial.tsv --output lex_preds/SICK/NLI/pred/
# torchrun scripts_NLI/2_SICK_NLI_predict.py --dataset SICK --output lex_preds/SICK/NLI/pred/ --part trial

import glob
import argparse
import csv
import re
import tqdm
import torch
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from torch.utils.data import DataLoader

np.int = np.int64
batchsize = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
parser.add_argument("--output", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--preds", required=False, default=0, metavar="FILES", help="Only give predictions")
args = parser.parse_args()
dataset = args.dataset
output = args.output
preds_bool = args.preds
part = args.part

model_name = "sileod/deberta-v3-base-tasksource-nli"
model_str = model_name.replace(r"/", "_")


def get_preds(csvreader, dataset_path, str_part):
    # to pipeline
    list_dict = []
    for row in iter(csvreader):
        list_dict.append(dict(prem=row["prem"], hyp=row["hyp"]))
    dataset_dict = Dataset.from_list(list_dict)


    def preprocess_function(rows, tokenizer):
        return tokenizer(rows["prem"], rows["hyp"], truncation=True, padding="max_length", max_length=64, return_tensors='pt')

    os.makedirs(f"{output}/full", exist_ok=True)
    os.makedirs(f"{output}/inter", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()


    dataset_token = dataset_dict.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer},)
    dataset_token = dataset_token.remove_columns(["prem", "hyp"])
    dataset_token = dataset_token.with_format("torch")
    dataloader = DataLoader(dataset_token, batch_size=batchsize)


    model_dict = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    preds = []
    probs = []

    softmax = torch.nn.Softmax(dim=-1)

    # probabilities = softmax(torch.from_numpy(predicciones.predictions))
    # probabilities_max = torch.max(probabilities, axis=1)
    # dataframe_inp["prob"] = pd.Series(probabilities_max.values.numpy())
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
                    out = model(i_ids, a_mask, t_ids)["logits"]
                else:
                    out = model(i_ids, a_mask)["logits"]
                preds += out.tolist()
                probs += softmax(out).tolist()

    if preds_bool:
        pd_train = pd.read_csv(dataset, sep="\t")
        pd_train["preds"] = preds
        pd_train.to_csv(f"{output}/full/predicts_{str_part}.tsv", sep="\t", index=None)

    else:
        result = pd.DataFrame(preds)
        result.to_csv(f"{output}/full/predicts_{str_part}_preds.tsv", sep="\t")
        result = pd.DataFrame(probs)
        result.to_csv(f"{output}/full/predicts_{str_part}_probs.tsv", sep="\t")

        
        result = pd.read_csv(dataset_path, delimiter="\t")
        result["preds"] = preds
        result["probs"] = probs

        res_group = result.groupby("CombID").agg({
            'head': lambda x: list(x)[0], 'tail': lambda x: list(x)[0],
            'preds': 'sum', 'probs': 'sum'
            })

        res_group[["head", "tail", "preds", "probs"]].to_csv(f"{output}/inter/predictions_{str_part}.tsv", sep="\t")

        # try:
        # except KeyError:
        #     # res_group = result.groupby("ID").agg({'preds': 'sum', 'label': lambda x: list(x)[0]})
        #     res_group = result.groupby("ID").agg({'preds': 'sum', 'probs': 'sum'})
        #     res_group = res_group[["label", "preds", "probs"]]
        #     res_group[["preds", "label", "probs"]].to_csv(f"{output}/inter/predictions_{part}.tsv", sep="\t")


dir_path = str(os.path.dirname(os.path.realpath(__file__)))
if part == "all":
    files_found = glob.glob(f"{dir_path}/../lex_preds/{dataset}/NLI/templates/*.tsv")
    for i in files_found:
        str_part = re.findall("templates/([A-z]*).tsv", i)[0]
        print(str_part)
        dataset_NLI = open(i, "r")
        csvreader = csv.DictReader(dataset_NLI, delimiter="\t")
        get_preds(csvreader, i, str_part)

else:    
    files_found = glob.glob(f"{dir_path}/../lex_preds/{dataset}/NLI/templates/{part}.tsv")
    dataset_NLI = open(files_found[0], "r")
    csvreader = csv.DictReader(dataset_NLI, delimiter="\t")
    get_preds(csvreader, files_found[0], part)
