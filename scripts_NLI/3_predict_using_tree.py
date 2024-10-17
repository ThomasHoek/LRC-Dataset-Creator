import argparse
import pickle
import pandas as pd
import glob
from ast import literal_eval
import os

from regex import F

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
        

def to_file(pred_file):
    NLI_word_info = pd.read_csv(f"lex_pairs/{dataset}/{dataset}_{part}_ccg.tsv", delimiter="\t")
    NLI_test = pd.read_csv(pred_file, delimiter="\t")
    test_x = list(NLI_test["preds"])
    test_x = [literal_eval(x) for x in test_x]
    NLI_word_info["pred"] = clf.best_estimator_.predict(test_x)

    os.makedirs(f"lex_KB/{dataset}/NLI/predictions", exist_ok=True)
    os.makedirs(f"lex_KB/{dataset}/NLI/final", exist_ok=True)

    NLI_word_info[["W1", "W2", "pred"]].to_csv(f"lex_KB/{dataset}/NLI/predictions/{part}.tsv", sep="\t")

    final = NLI_word_info.apply(get_prolog_sen, axis=1)
    final.dropna(inplace=True)
    final.to_csv(f'lex_KB/{dataset}/NLI/final/{part}.pl', sep='\n', index=False, header=False)

    final_lower = NLI_word_info.apply(lambda x: get_prolog_sen(x, lemma=True, index=False), axis=1)
    final_lower.dropna(inplace=True)
    final_lower.to_csv(f'lex_KB/{dataset}/NLI/final/{part}_lemma.pl', sep='\n', index=False, header=False)

    final_lower = NLI_word_info.apply(lambda x: get_prolog_sen(x, lemma=True, index=True), axis=1)
    final_lower.dropna(inplace=True)
    final_lower.to_csv(f'lex_KB/{dataset}/NLI/final/{part}_lemma_idx.pl', sep='\n', index=False, header=False)
    # break


if part == "all":
    files_found = glob.glob(f"lex_preds/{dataset}/NLI/inter/*.tsv")
    for i in files_found:
        to_file(i)

else:    
    files_found = glob.glob(f"lex_preds/{dataset}/NLI/pred/inter/predictions_{part}.tsv")
    to_file(files_found[0])
