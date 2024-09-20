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

    ("neutral", "neutral"): "disjoint",

    ("entailment", "contradiction"): "independent",
    ("neutral", "contradiction"): "independent",
    ("contradiction", "entailment"): "independent",
    ("contradiction", "neutral"): "independent",
}

priority_prediction = [
    ("entailment", "entailment"), ("neutral", "neutral"), ("contradiction", "contradiction"),
    ("entailment", "neutral"), ("neutral", "entailment"), ("neutral", "contradiction"), ("contradiction", "neutral"),
    ("entailment", "contradiction"), ("contradiction", "entailment")
    ]


def priority_choice(group: list[Any]) -> Any:
    assert all(x in priority_prediction for x in group)
    return max(group, key=lambda name: -priority_prediction.index(name))


def to_solution(working_dir: str, dataset: str, part: str, pred_file: str, template_check: bool = True, label_bool: bool = False):
    template_dict: dict[str, list[tuple[str, str]]] = {}
    prediction_dict: dict[int, Counter[tuple[str, str]]] = {}
    solution_dict: dict[str, str] = {}

    model_name = re.search(f"{working_dir}/Results/{dataset}/NLI/(.*)/{part}.tsv", pred_file)
    assert model_name is not None
    model_name: str = model_name.group(1)

    os.makedirs(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results", exist_ok=True)
    sol_file = f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/preds.tsv"
    if label_bool:
        counter_file = f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/count_preds.txt"
    count_solutions_dict: dict[str, Counter[tuple[str, str]]] = {}

    pred_csv = pd.read_csv(pred_file, delimiter="\t")
    grouped = pred_csv.groupby(["CombID", "SenID", "templatenum"])

    # CombID	SenID	templatenum	prem	hyp	label	top1	top1_score	top2	top2_score	top3	top3_score
    for _, group in grouped:
        # first sentence
        first: Any = group.iloc[0]
        sen_num_1: str = first["CombID"]
        problemid_1: str = first["SenID"]
        template_num1: str = first["templatenum"]
        pred_1: str = first["top1"]

        # second sentence
        second: Any = group.iloc[1]
        sen_num_2: str = second["CombID"]
        problemid_2: str = second["SenID"]
        template_num2: str = second["templatenum"]
        pred_2: str = second["top1"]

        if label_bool:
            label_1: str = first["label"]
            label_2: str = second["label"]
            assert label_1 == label_2

        assert sen_num_1 == sen_num_2
        assert problemid_1 == problemid_2
        assert template_num1 == template_num2

        pred_1 = pred_1.lower()
        pred_2 = pred_2.lower()

        #  write for each template
        if template_check:
            if template_num1 not in template_dict:
                template_dict[template_num1] = []
            template_dict[template_num1].append((pred_1, pred_2))

        # normal way
        if int(sen_num_1) not in prediction_dict:
            prediction_dict[int(sen_num_1)] = Counter()

            if label_bool:
                solution_dict[int(sen_num_1)] = label_1

        # other way to determine prediction
        # prediction_dict[int(problemid_1)].update([pred_dict[(pred_1, pred_2)]])
        prediction_dict[int(sen_num_1)].update([(pred_1, pred_2)])

        if label_bool:
            if label_1 not in count_solutions_dict:
                count_solutions_dict[label_1] = Counter()
    
            count_solutions_dict[label_1].update([(pred_1, pred_2)])

    # print(prediction_dict)
    # exit()
    if label_bool:
        counter_file_r = open(counter_file, "w+")
        for key in count_solutions_dict:
            counter_file_r.write(f"{key}\n")
            for i in count_solutions_dict[key].most_common():
                counter_file_r.write(f"{i}\n")
            counter_file_r.write(f"{'-'*25}\n")
        counter_file_r.close()

    if template_check and label_bool:
        os.makedirs(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/template", exist_ok=True)
        real = list(solution_dict.values())
        for i in list(template_dict.keys()):
            preds = [pred_dict[x] for x in template_dict[i]]
            with open(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/template/{i}.tsv", "w+") as sol_file_template:
                for pred, lab in zip(preds, real):
                    sol_file_template.write(f"{lab}\t{pred}\n")

            for x in template_dict:
                assert len(template_dict[x]) == len(real)
                cr = classification_report(real, y_pred=preds, zero_division=np.nan)
                cr_file = open(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/template/{i}_classreport.txt", "w+")
                cr_file.write(cr)
                cr_file.close()

                # cm = confusion_matrix()
                disp = ConfusionMatrixDisplay.from_predictions(real, preds)
                disp.plot().figure_.savefig(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/template/{i}_confmatrix.png")
                plt.close()

    #################################################

    all_preds = pd.read_csv(pred_file, delimiter="\t")
    preds = []
    for pred_counter_i in prediction_dict:
        pred_counter = prediction_dict[pred_counter_i]
        m = max(pred_counter.values())
        r = [k for k in pred_counter if pred_counter[k] == m]
        preds.append(pred_dict[priority_choice(r)])

    if label_bool:
        real = list(solution_dict.values())
        assert len(real) == len(preds)

        cr = classification_report(real, y_pred=preds, zero_division=np.nan)
        cr_file = open(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/classreport.txt", "w+")
        cr_file.write(cr)
        cr_file.close()

        # cm = confusion_matrix()
        disp = ConfusionMatrixDisplay.from_predictions(real, preds)
        disp.plot().figure_.savefig(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/confmatrix.png")
        plt.close()

        #  add extra debug for wrong entrees
        wrong_preds = [c for (c,r),p in zip(solution_dict.items(), preds) if r != p]

        with open(sol_file, "w+") as sol_write:
            sol_write.write(f"Counter\tWord1\tWord2\tReal\tPrediction\n")
            for c, p, r in zip(prediction_dict.keys(), preds, real):
                pred_series = all_preds[all_preds["CombID"] == c].iloc[-1]
                sol_write.write(f"{c}\t{pred_series['prem']}\t{pred_series['hyp']}\t{r}\t{p}\n")

        # all_preds[all_preds["problemid"].isin()].to_csv(f"ccg_phrases/PPDB_res/{model_name}/wrong.csv", sep="\t", header=None)
        with open(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/wrong.csv", "w+") as wrong_file:
            for wrong_number in wrong_preds:
                wrong_group = all_preds[all_preds["CombID"] == wrong_number]
                for _, row in wrong_group.iterrows():
                    wrong_file.write(f"{wrong_number}\t{row['prem']}\t{row['hyp']}\t{row['label']}\t{row['top1']}\t{str(prediction_dict[wrong_number])}\n")

        # all_preds[all_preds["problemid"].isin()].to_csv(f"ccg_phrases/PPDB_res/{model_name}/wrong.csv", sep="\t", header=None)
        correct_preds = [c for (c,r),p in zip(solution_dict.items(), preds) if r == p]

        with open(f"{working_dir}/Results/{dataset}/NLI/{model_name}/results/correct.csv", "w+") as correct_file:
            for correct_number in correct_preds:
                correct_group = all_preds[all_preds["CombID"] == correct_number]
                for _, row in correct_group.iterrows():
                    correct_file.write(f"{correct_number}\t{row['prem']}\t{row['hyp']}\t{row['label']}\t{row['top1']}\t{str(prediction_dict[correct_number])}\n")
    else:
        with open(sol_file, "w+") as sol_write:
            sol_write.write(f"Counter\tWord1\tWord2\tPrediction\n")
            for c, p in zip(prediction_dict.keys(), preds):
                pred_series = all_preds[all_preds["CombID"] == c].iloc[-1]
                sol_write.write(f"{c}\t{pred_series['prem']}\t{pred_series['hyp']}\t{p}\n")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
    args = parser.parse_args()
    dataset = args.dataset
    part = args.part

    root_dir = str(os.path.dirname(os.path.realpath(__file__))) + "/.."
    print(root_dir)

    pred_files = glob.glob(f"{root_dir}/Results/{dataset}/NLI/*/{part}.tsv")
    assert pred_files

    # datasets with labels
    for i in pred_files:
        to_solution(root_dir, dataset, part, i, True, True)
