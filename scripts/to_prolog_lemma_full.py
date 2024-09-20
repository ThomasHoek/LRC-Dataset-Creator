import argparse
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
cur_dir: str = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
parser.add_argument("--model", required=True, metavar="FILES", help="Part of dataset to test on")
parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
args = parser.parse_args()
dataset = args.dataset
model = args.model
part = args.part


# https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def get_prolog_sen(df, lower=False, lemma=False):
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

    if lemma:
        w1 = lemmatize_sentence(w1)
        w2 = lemmatize_sentence(w2)

    w1 = w1.replace("'", r"\'")
    w2 = w2.replace("'", r"\'")
    w1 = w1.replace(r"\\", "\\")
    w2 = w2.replace(r"\\", "\\")
    w1 = w1.replace(r"\ ", "")
    w2 = w2.replace(r"\ ", "")


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

final_lem = NLI_word_info[["W1", "W2", "pred"]].apply(get_prolog_sen, args=(False, True), axis=1)

    
final.dropna(inplace=True)
final_lem.dropna(inplace=True)

final = pd.concat([final, final_lem], ignore_index=True)
final.drop_duplicates(inplace=True)

os.makedirs(f'Results/{dataset}/prolog_{model}_lem_ful/', exist_ok=True)
final.to_csv(f'Results/{dataset}/prolog_{model}_lem_ful/{dataset}_{part}.pl', sep='\n', index=False, header=False)
# break

