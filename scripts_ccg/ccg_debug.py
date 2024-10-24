# splits a lex pair file into a seperate debugged file of each tag for debugging reasons.
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--filename", required=True, default=True, help="disable duplicates and write to meta file")
args = parser.parse_args()

dataset = args.dataset
part = args.filename

path_tsv = f"lex_pairs/{dataset}/{part}.tsv"
tag_file = pd.read_csv(path_tsv, sep="\t")

os.makedirs(f"lex_pairs/{dataset}/debug", exist_ok=True)

for (merge, tag_1, tag_2), df in tag_file.groupby(['merge_tag', 'W1_tag', 'W2_tag']):
    tag_1 = tag_1.replace('/', '_u_')
    tag_2 = tag_2.replace('/', '_u_')
    print(tag_1, tag_2)
    save_path  = rf"lex_pairs/{dataset}/debug/{merge}___{tag_1}_{tag_2}.tsv"
    df.to_csv(save_path, sep="\t")