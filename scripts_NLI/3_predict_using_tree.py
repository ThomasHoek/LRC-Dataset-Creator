import pickle
import pandas as pd
import glob
from ast import literal_eval
import os

model_dir = "models/NLI/tasksource_full/model.pkl"

with open(model_dir, 'rb') as f:
    clf = pickle.load(f)

pred_file_list = glob.glob("Results/SICK/task_source/inter/*.tsv")
os.makedirs("Results/SICK/task_source/predictions", exist_ok=True)
os.makedirs("Results/SICK/task_source/final", exist_ok=True)


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
    match df["pred"]:
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
            print(df["pred"])
            return
        


for pred_file in pred_file_list:
    part = pred_file.replace("predictions_", "").replace(".tsv", "").split("/")[-1]
    NLI_word_info = pd.read_csv(f"Results/SICK/SICK_{part}_ccg.tsv", delimiter="\t")
    NLI_test = pd.read_csv(pred_file, delimiter="\t")
    test_x = list(NLI_test["preds"])
    test_x = [literal_eval(x) for x in test_x]
    NLI_word_info["pred"] = clf.best_estimator_.predict(test_x)
    NLI_word_info[["W1", "W2", "pred"]].to_csv(f"Results/SICK/task_source/predictions/{part}.tsv", sep="\t")

    final = NLI_word_info[["W1", "W2", "pred"]].apply(get_prolog_sen, axis=1)
    final.dropna(inplace=True)
    final.to_csv(f'Results/SICK/task_source/final/SICK_{part}.pl', sep='\n', index=False, header=False)

    final_lower = NLI_word_info[["W1", "W2", "pred"]].apply(lambda x: get_prolog_sen(x, lower=True), axis=1)
    final_lower.dropna(inplace=True)
    final_lower.to_csv(f'Results/SICK/task_source/final/SICK_{part}_lower.pl', sep='\n', index=False, header=False)
    # break
