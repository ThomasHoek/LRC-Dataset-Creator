from __future__ import annotations  # typing fix??
import re
import argparse
import pickle
import pandas as pd
import glob
from ast import literal_eval
import os


model_dir = "models/NLI/tasksource_full/model.pkl"

with open(model_dir, 'rb') as f:
    clf = pickle.load(f)


parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
args = parser.parse_args()
dataset = args.dataset
part = args.part


def get_prolog_sen(df, lower=False, lemma=False, index=False):
    """
    get_prolog_sen transforms dataframe to prolog

    Uses the word pairs and the prediction to translate it into a prolog relation for Langpro.
    Adhered to the template: ind_rel(X(W1,W2)).
    Independent cases are ignored.
    Where X indicated:
        isa_wn: Hyponym
        ant_wn: Antonym (not used)
        der_wn -> Der?
        sim_wn: Synonym
        disj: Disjoint

    Args:
        df (DataFrame): Results and predictions dataframe
    """
    if lemma:
        w1 = df["L1"]
        w2 = df["L2"]
    else:
        w1 = df["W1"]
        w2 = df["W2"]

    if lower:
        w1 = w1.lower()
        w2 = w2.lower()
        
    match df["pred"]:
        case "disjoint":
            final_str = f"ind_rel(disj('{w1}','{w2}'))."
        case "forwardentailment":
            final_str = f"ind_rel(isa_wn('{w1}','{w2}'))."
        case "reverseentailment":
            final_str = f"ind_rel(isa_wn('{w2}','{w1}'))."
        case "synonym":
            final_str = f"ind_rel(sim_wn('{w1}','{w2}'))."
        case "independent":
            return 
        case _:
            print(df["pred"])
            return

    if index:
        final_str = final_str.replace("')).", f"','{df['ProbID']}')).")
    return final_str


def add_duplicates(meta_file: str, main_df: pd.DataFrame, existing_series: pd.Series[str]):
    import json
    with open(meta_file) as f:
        duplicates = json.load(f)

    for dup in duplicates:
        duplicate_list = duplicates[dup]
        problem: pd.DataFrame = main_df[main_df["ProbID"] == duplicate_list[0]]

        w1_org, w2_org = dup.split("_*_")
        w1_org = w1_org.replace("+=+", " ")
        w2_org = w2_org.replace("+=+", " ")

        problem = problem[problem["W1"] == w1_org]
        problem = problem[problem["W2"] == w2_org]

        # should be Series now, but still in DF format.
        assert len(problem) == 1

        # get lemma
        w1: str = problem["L1"].item()
        w2: str = problem["L2"].item()

        match problem["pred"].item():
            case "disjoint":
                final_str = f"ind_rel(disj('{w1}','{w2}'))."
            case "forwardentailment":
                final_str = f"ind_rel(isa_wn('{w1}','{w2}'))."
            case "reverseentailment":
                final_str = f"ind_rel(isa_wn('{w2}','{w1}'))."
            case "synonym":
                final_str = f"ind_rel(sim_wn('{w1}','{w2}'))."
            case "independent":
                continue
            case _:
                # print if things go wrong???
                # should NEVER happen
                print(problem["pred"].item())
                continue

        final_duplicate_lst: list[str] = []
        for dup_number in duplicate_list[1:]:
            final_duplicate_lst.append(final_str.replace("')).", f"','{dup_number}'))."))
        existing_series = pd.concat([existing_series, pd.Series(final_duplicate_lst)])
    return existing_series


def make_files(word_info_df: pd.DataFrame, prediction_pd: pd.Series[str], str_part: str):    
    test_x = list(prediction_pd["preds"])
    test_x_eval: list[float] = [literal_eval(x) for x in test_x]

    word_info_df["pred"] = clf.best_estimator_.predict(test_x_eval)

    os.makedirs(f"lex_KB/{dataset}/NLI/predictions", exist_ok=True)
    os.makedirs(f"lex_KB/{dataset}/NLI/final", exist_ok=True)

    word_info_df[["W1", "W2", "pred"]].to_csv(f"lex_KB/{dataset}/NLI/predictions/{part}.tsv", sep="\t")

    final = word_info_df.apply(get_prolog_sen, axis=1)
    final.dropna(inplace=True)
    final.drop_duplicates(inplace=True)
    final.to_csv(f'lex_KB/{dataset}/NLI/final/{str_part}.pl', sep='\n', index=False, header=False)

    final_lemma = word_info_df.apply(lambda x: get_prolog_sen(x, lemma=True, index=False), axis=1)
    final_lemma.dropna(inplace=True)
    final_lemma.to_csv(f'lex_KB/{dataset}/NLI/final/{str_part}_lemma.pl', sep='\n', index=False, header=False)

    final_lemma_idx = word_info_df.apply(lambda x: get_prolog_sen(x, lemma=True, index=True), axis=1)
    final_lemma_idx = add_duplicates(f"lex_pairs/{dataset}/meta/{dataset}_{str_part}_ccg.json", word_info_df, final_lemma_idx)
    final_lemma_idx.dropna(inplace=True)
    final_lemma_idx.to_csv(f'lex_KB/{dataset}/NLI/final/{str_part}_lemma_idx.pl', sep='\n', index=False, header=False)
    # break


if part == "all":
    all_list = glob.glob(f"lex_preds/{dataset}/NLI/pred/inter/*.tsv")
    for file_part in all_list:
        print(file_part)
        str_part = re.findall("predictions_([A-z]*).tsv", file_part)[0]

        NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{dataset}_{str_part}_ccg.tsv", delimiter="\t")
        pred_df = pd.read_csv(file_part, delimiter="\t")
        make_files(NLI_word_info, pred_df, str_part)    

else:
    pred_df = pd.read_csv(f"lex_preds/{dataset}/NLI/pred/inter/predictions_{part}.tsv", delimiter="\t")
    NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{dataset}_{part}_ccg.tsv", delimiter="\t")
    make_files(NLI_word_info, pred_df, part)
