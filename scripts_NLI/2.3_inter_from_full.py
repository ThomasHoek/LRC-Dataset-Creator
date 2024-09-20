# This script is used if an error was found during training.
import argparse
import pandas as pd
import math

parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--part", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--output", required=True, metavar="FILES", help="Dataset to test on")
args = parser.parse_args()
output = args.output
part = args.part

result = pd.read_csv(f"{output}/full/predicts_{part}.tsv", delimiter="\t", names=["idx", 0, 1, 2], skiprows=1)
inter = open(f"{output}/inter/predicts_{part}.tsv", "w+")

def row_to_list(df_group):
    num_list = []
    for _, row in df_group.iterrows():
        for i in range(3):
            num_list.append(row[i])

    assert len(num_list) == (18 * 3) 
    return num_list

inter.write("CombID\tpreds\n")
for idx, group in result.groupby(result.index // 18):
    row_out = row_to_list(group)
    inter.write(f"{idx}\t{row_out}\n")

