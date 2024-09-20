import glob
import re
import os
from typing import Any
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pred_dict = {
    ("entailment", "entailment"): "synonym",
    ("contradiction", "contradiction"): "alternation",
    # entail
    ("entailment", "neutral"): "forwardentailment",
    ("neutral", "entailment"): "reverseentailment",
    # disjoint ????
    ("neutral", "neutral"): "disjoint",
    # other relations ???
    ("entailment", "contradiction"): "independent",
    ("neutral", "contradiction"): "independent",
    ("contradiction", "entailment"): "independent",
    ("contradiction", "neutral"): "independent",
    # TODO: check mapping_template.txt
}

priority_prediction = [
    ("entailment", "entailment"),
    ("neutral", "neutral"),
    ("contradiction", "contradiction"),
    ("entailment", "neutral"),
    ("neutral", "entailment"),
    ("neutral", "contradiction"),
    ("contradiction", "neutral"),
    ("entailment", "contradiction"),
    ("contradiction", "entailment"),
]


def priority_choice(group: list[Any]) -> Any:
    assert all(x in priority_prediction for x in group)
    return max(group, key=lambda name: -priority_prediction.index(name))


def to_solution(working_dir: str, dataset: str, part: str, pred_file: str, template_check: bool = True, label_bool: bool = False):
    prediction_dict: dict[int, Counter[tuple[str, str]]] = {}

    # make dirs
    os.makedirs(f"{working_dir}/Results/{dataset}/NLI/results", exist_ok=True)
    sol_file = f"{working_dir}/Results/{dataset}/NLI/results/preds.tsv"

    # get files
    pred_csv = pd.read_csv(pred_file, delimiter="\t")
    grouped = pred_csv.groupby(["CombID", "templatenum"])

    # CombID	SenID	templatenum	prem	hyp	label	top1	top1_score	top2	top2_score	top3	top3_score
    for _, group in grouped:
        # first sentence
        first: Any = group.iloc[0]
        sen_num_1: str = first["CombID"]
        # problemid_1: str = first["SenID"]
        template_num1: str = first["templatenum"]
        pred_1: str = first["preds"]

        # second sentence
        second: Any = group.iloc[1]
        sen_num_2: str = second["CombID"]
        # problemid_2: str = second["SenID"]
        template_num2: str = second["templatenum"]
        pred_2: str = second["preds"]

        assert sen_num_1 == sen_num_2
        # assert problemid_1 == problemid_2
        assert template_num1 == template_num2

        pred_1 = pred_1.lower()
        pred_2 = pred_2.lower()

        if int(sen_num_1) not in prediction_dict:
            prediction_dict[int(sen_num_1)] = Counter()
        prediction_dict[int(sen_num_1)].update([(pred_1, pred_2)])

    all_preds = pd.read_csv(pred_file, delimiter="\t")
    preds = []
    for pred_counter_i in prediction_dict:
        pred_counter = prediction_dict[pred_counter_i]
        m = max(pred_counter.values())
        r = [k for k in pred_counter if pred_counter[k] == m]
        preds.append(pred_dict[priority_choice(r)])

    with open(sol_file, "w+") as sol_write:
        sol_write.write(f"Counter\tWord1\tWord2\tPrediction\n")
        for c, p in zip(prediction_dict.keys(), preds):
            # take shortest
            pred_series = all_preds[all_preds["CombID"] == c].iloc[-1]
            sol_write.write(f"{c}\t{pred_series['prem'][2:]}\t{pred_series['hyp'][2:]}\t{p}\n")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
    args = parser.parse_args()
    dataset = args.dataset
    part = args.part

    root_dir = str(os.path.dirname(os.path.realpath(__file__))) + "/../.."
    print(root_dir)

    pred_files = glob.glob(f"{root_dir}/Results/{dataset}/*/predicts_{part}.tsv")
    assert pred_files

    # datasets with labels
    for i in pred_files:
        to_solution(root_dir, dataset, part, i, False, False)
