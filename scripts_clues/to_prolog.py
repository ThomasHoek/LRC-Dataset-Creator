import argparse
import pandas as pd
import os

cur_dir: str = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--model", required=True, metavar="FILES", help="Part of dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
args = parser.parse_args()
dataset = args.dataset
model = args.model
part = args.part


def get_prolog_sen(df, lower=False):
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
    w1 = df["W1"]
    w2 = df["W2"]
    if lower:
        w1 = w1.lower()
        w2 = w2.lower()

    w1 = w1.replace("'", r"\'")
    w2 = w2.replace("'", r"\'")
    w1 = w1.replace(r"\\", "\\")
    w2 = w2.replace(r"\\", "\\")
    match df["pred"]:
        case "disjoint":
            return f"ind_rel(disj('{w1}','{w2}'))."
        case "forwardentailment" | "forward_entailment" | "hyper":
            return f"ind_rel(isa_wn('{w1}','{w2}'))."
        case "reverseentailment":
            return f"ind_rel(isa_wn('{w2}','{w1}'))."
        case "synonym" | "equivalence":
            return f"ind_rel(sim_wn('{w1}','{w2}'))."
        case "antonym" | "alternation":
            return f"ind_rel(ant_wn('{w1}','{w2}'))."
        case "independent" | "random":
            return 
        case _:
            print(df["pred"])
            return

try:
    NLI_word_info = pd.read_csv(f"Results/{dataset}/{model}/predicts_{part}.tsv", delimiter="\t")
    final = NLI_word_info[["W1", "W2", "pred"]].apply(get_prolog_sen, axis=1)
except KeyError:
    NLI_word_info = pd.read_csv(f"Results/{dataset}/{model}/predicts_{part}.tsv", delimiter="\t", names=["W1", "W2", "pred"])
    final = NLI_word_info[["W1", "W2", "pred"]].apply(get_prolog_sen, axis=1)
final.dropna(inplace=True)
os.makedirs(f'Results/{dataset}/prolog_{model}/', exist_ok=True)
final.to_csv(f'Results/{dataset}/prolog_{model}/{dataset}_{part}.pl', sep='\n', index=False, header=False)
# break
