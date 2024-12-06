from __future__ import annotations  # typing fix??
import argparse
import glob
import re
import pandas as pd
import os


cur_dir: str = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="File used to create a prolog file from lex_preds.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
parser.add_argument("--model", required=True, metavar="FILES", help="Which model was used for predictions")
args = parser.parse_args()
dataset = args.dataset
model = args.model
part = args.part


def get_relation(pred: str, w1: str, w2: str):
    match pred:
        case "disjoint":
            return f"ind_rel(disj('{w1}','{w2}'))."
        case "forwardentailment":
            return f"ind_rel(isa_wn('{w1}','{w2}'))."
        case "reverseentailment":
            return f"ind_rel(isa_wn('{w2}','{w1}'))."
        case "synonym":
            return f"ind_rel(sim_wn('{w1}','{w2}'))."
        case "independent":
            return 
        case _:
            print(pred)
            return


def get_prolog_sen(df: pd.Series[str], lower: bool = False, lemma: bool = False, index: bool = False):
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

    final_str = get_relation(df["pred"], w1, w2)

    if index and final_str:
        final_str = final_str.replace("')).", f"'),{df['ProbID']}).")
    return final_str


def add_duplicates(duplicates: dict[str, list[int]], main_df: pd.DataFrame, existing_series: pd.Series[str]):
    final_duplicate_lst: list[str] = []
    for dup in duplicates:
        duplicate_list = duplicates[dup]

        w1_org, w2_org = dup.split("_*_")
        w1_org = w1_org.replace("+=+", " ")
        w2_org = w2_org.replace("+=+", " ")

        problem: pd.DataFrame = main_df[main_df.ProbID == duplicate_list[0]]
        problem = problem[(problem.W1 == w1_org) & (problem.W2 == w2_org)]
        # should be Series now, but still in DF format.
        if len(problem) != 1:
            print(dup, duplicate_list)
            continue

        # convert to series hack
        problem = problem.iloc[0]

        # get lemma
        w1: str = problem["L1"]
        w2: str = problem["L2"]

        final_str = get_relation(problem["pred"], w1, w2)

        if final_str:
            for dup_number in duplicate_list[1:]:
                final_duplicate_lst.append(final_str.replace("')).", f"'),{dup_number})."))
    existing_series = pd.concat([existing_series, pd.Series(final_duplicate_lst)], axis=0, sort=False)
    return existing_series


def make_files(word_info_df: pd.DataFrame, prediction_df: pd.DataFrame, str_part: str):
    word_info_df["pred"] = prediction_df["pred"]
    os.makedirs(f'lex_preds/{dataset}/{model}/pred/', exist_ok=True)
    word_info_df.to_csv(f'lex_preds/{dataset}/{model}/pred/{str_part}.tsv', sep='\t')

    os.makedirs(f"lex_KB/{dataset}/{model}/predictions", exist_ok=True)
    os.makedirs(f"lex_KB/{dataset}/{model}/final", exist_ok=True)

    word_info_df[["W1", "W2", "pred"]].to_csv(f"lex_KB/{dataset}/{model}/predictions/{str_part}.tsv", sep="\t")
    final = word_info_df.apply(get_prolog_sen, axis=1)

    print("normal")
    final.dropna(inplace=True)
    final.drop_duplicates(inplace=True)
    final.to_csv(f'lex_KB/{dataset}/{model}/final/{str_part}.pl', sep='\n', index=False, header=False)

    print("lemma")
    final_lemma = word_info_df.apply(lambda x: get_prolog_sen(x, lemma=True, index=False), axis=1)
    final_lemma.dropna(inplace=True)
    final_lemma.to_csv(f'lex_KB/{dataset}/{model}/final/{str_part}_lemma.pl', sep='\n', index=False, header=False)

    print("lemma_idx")
    final_lemma_idx = word_info_df.apply(lambda x: get_prolog_sen(x, lemma=True, index=True), axis=1)
    import json
    try:
        with open(f"lex_pairs/{dataset}/meta/{dataset}_{str_part}_ccg.json") as f:
            duplicates = json.load(f)
    except FileNotFoundError:
        with open(f"lex_pairs/{dataset}/meta/{str_part}_ccg.json") as f:
            duplicates = json.load(f)
        
    assert duplicates
    final_lemma_idx = add_duplicates(duplicates, word_info_df, final_lemma_idx)
    final_lemma_idx.dropna(inplace=True)
    final_lemma_idx.to_csv(f'lex_KB/{dataset}/{model}/final/{str_part}_lemma_idx.pl', sep='\n', index=False, header=False)
    # break


if part == "all":
    all_list = glob.glob(f"lex_preds/{dataset}/{model}/predicts_*.tsv")
    for file_part in all_list:
        print(file_part)
        str_part = re.findall("predicts_([A-z1-9]*).tsv", file_part)[0]

        try:
            NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{dataset}_{str_part}_ccg.tsv", delimiter="\t")
        except FileNotFoundError:
            NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{str_part}_ccg.tsv", delimiter="\t")
        pred_df = pd.read_csv(file_part, delimiter="\t")
        make_files(NLI_word_info, pred_df, str_part)    

else:
    pred_df = pd.read_csv(f"lex_preds/{dataset}/{model}/predicts_{part}.tsv", delimiter="\t")
    try:
        NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{dataset}_{part}_ccg.tsv", delimiter="\t")
    except FileNotFoundError:
        NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{part}_ccg.tsv", delimiter="\t")
    make_files(NLI_word_info, pred_df, part)


# if part == "all":
#     all_list = glob.glob(f"lex_preds/{dataset}/{model}/predicts_*.tsv")
#     for file_part in all_list:
#         print(file_part)
#         str_part = re.findall("predictions_([A-z1-9]*).tsv", file_part)[0]
#         NLI_word_info = pd.read_csv(file_part, delimiter="\t")
#         make_files(NLI_word_info, str_part)    

# else:
#     NLI_word_info = pd.read_csv(f"lex_preds/{dataset}/{model}/predicts_{part}.tsv", delimiter="\t")
#     make_files(NLI_word_info, part)

