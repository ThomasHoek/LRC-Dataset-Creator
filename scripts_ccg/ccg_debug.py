from pandas import DataFrame, Series   # typing
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description="Splits a lex_pred or lex_pair file into a seperate debugged file of each tag.")
parser.add_argument("--pred_or_pair", required=True, metavar="FILES", help="choice for Lex_Pred or Lex_Pair to split on")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, default=True, help="Name of the file")


args = parser.parse_args()
pred_or_pair = "preds" if "pred" in args.pred_or_pair else "pairs"
dataset = args.dataset
part = args.part

dir_path = str(os.path.dirname(os.path.realpath(__file__)))

tag_file: DataFrame
# get correct path
if pred_or_pair == "preds":
    path_tsv = f"{dir_path}/../lex_{pred_or_pair}/{dataset}/NLI/pred/{part}.tsv"
    tag_file = pd.read_csv(path_tsv, sep="\t")
else:
    path_tsv = f"{dir_path}/../lex_{pred_or_pair}/{dataset}/{part}_ccg.tsv"
    tag_file = pd.read_csv(path_tsv, sep="\t")

os.makedirs(f"lex_{pred_or_pair}/{dataset}/debug", exist_ok=True)

# split by tags
# merge: Series[str]
# tag_1: Series[str]
# tag_2: Series[str]

for (merge, tag_1, tag_2), df in tag_file.groupby(['merge_tag', 'W1_tag', 'W2_tag']):
    tag_1 = tag_1.replace('/', '_u_')
    tag_2 = tag_2.replace('/', '_u_')
    save_path = rf"{dir_path}/../lex_{pred_or_pair}/{dataset}/debug/{merge}___{tag_1}_{tag_2}.tsv"
    df.to_csv(save_path, sep="\t")
