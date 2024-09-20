from copy import deepcopy
import re

from numpy import right_shift
import ccg_parse
from ccg_class import tree, leaf
from typing import Callable
import csv
import os

# import iso3166
# FIXME: dont filter country NERS?

# FIXME : lowercase check
#  update to compatible list
word_list = {
    r"JJ": "phrasal",

    # 2.	NN	Noun, singular or mass
    r"NN": "NP",
    # 13.	NNS	Noun, plural
    r"NNS": "NP",
    # 14.	NNP	Proper noun, singular
    r"NNP": "NP",
    # 15.	NNPS	Proper noun, plural
    r"NNPS": "NP",

    # # ------- Change to VP | TODO: test effect of VP
    # 27.	VB	Verb, base form
    r"VB": "VP",
    # 28.	VBD	Verb, past tense
    r"VBD": "VP",
    # 29.	VBG	Verb, gerund or present participle
    r"VBG": "VP",
    # 30.	VBN	Verb, past participle
    r"VBN": "VP",
    # 31.	VBP	Verb, non-3rd person singular present
    r"VBP": "VP",
    # 32.	VBZ	Verb, 3rd person singular present
    r"VBZ": "VP",
}

# acceptable parents for N's
# TODO: allow NP, but not as LAST parent. Do double parent check if NP -> if typeraised back to N or used conj 
np_parent_lst = [r"n", r"n/n", "n/pp", r"n\n", "n\n", "n/n"]
verb_reject_lst = ["is", "was", "be", "have", "been", "were", "are"]

#  Subtree lists
tree_tags = {r"IN": "phrasal"}
ccg_allowed = {r"n": "NP"}

# Original snipet from country info.
# country_long_lst: list[str] = list(iso3166.countries_by_name) + list(iso3166.countries_by_alpha2) + list(iso3166.countries_by_alpha3)


def to_tree(ccg_inp: list[str]) -> dict[int, tree]:
    """Parses list of CCG strings into a dict of trees. Dict key is CCG num."""
    ccg_data: list[list[tuple[int, str]]] = ccg_parse.parse_data(ccg_inp)
    return ccg_parse.parse_class(ccg_data)


def type_mix(type_1: str, type_2: str, allow_mix: bool = False) -> bool:
    if allow_mix:
        # Rules for extra allowed mixes | Used for PPDB
        combs = [("VP", "NP"), ("JJ", "NP"), ("JJ", "VP"), ("phrasal", "VP"), ("phrasal", "NP"), ("phrasal", "JJ")]
        return (type_1, type_2) in combs or (type_2, type_1) in combs
    return False


# simplified, add plural + clean options back later from hidden script.
def tree_to_phrase(tree_inp: tree) -> list[tuple[str, str, str]]:
    """
    tree_to_phrase Find every interesting phrase inside a CCG tree.

    Args:
        tree_inp (tree): CCG tree

    Returns:
        list[tuple[str, str, str]: list of (merge category, specific category, sentence)
    """
    global dataset
    collected: list[tuple[str, str, str]] = []  # | tuple[str, str, bool, str]] = []
    # reject_lst = ["'", "'", "\\'", r"'", r"\'", "’"]
    # reject_extra: Callable[[leaf], bool] = lambda x_lamb: x_lamb.word in reject_lst

    # TODO: add countries back | add information for country synnonyms???
    reject_per: Callable[[leaf], bool] = lambda x_lamb: x_lamb.BIO_ner == "I-PER"
    reject_org: Callable[[leaf], bool] = lambda x_lamb: x_lamb.BIO_ner == "I-ORG"
    reject_loc: Callable[[leaf], bool] = lambda x_lamb: x_lamb.BIO_ner == "I-LOC"
    BIO_NER_reject = ["I-PER", "I-ORG", "I-LOC"]

    # if ANY not in np list, but allow if CONJ
    #  TODO: bettter CONJ check ->
    # child_check: Callable[[tree], bool] = lambda x_lamb: x_lamb.syn_type not in np_parent_lst and x_lamb.combinator != "conj"
    child_check: Callable[[tree], bool] = lambda x_lamb: x_lamb.syn_type not in np_parent_lst 

    # FIXME: config file
    max_size = 4

    extra_subtree: dict[tree, str] = {}
    # ========= LEAVES =========
    for word in tree_inp.get_leaves([]):
        if word.POS in word_list.keys() and word.BIO_ner not in BIO_NER_reject:
            collected.append((word_list[word.POS], word.POS, word.word.strip()))

        # weird fix for IN parts
        if word.POS in tree_tags:
            parent_tree = word.parent
            while parent_tree.parent_check() and parent_tree.parent.syn_type in np_parent_lst:
                parent_tree = parent_tree.get_parent_tree()
            extra_subtree[parent_tree] = tree_tags[word.POS]

    # ========= TREES =========
    for x in tree_inp.gen_subtrees():
        if x.length_check(max_size):
            continue

        # TODO: properly implement a parent check with theory. For now keep all.
        # TODO 2: allow ALL, but add HEAD disallow. If det found as head -> remove.
        # check if no outer
        if x.parent_check() and x not in extra_subtree:
            if x.parent.syn_type in ccg_allowed and x not in extra_subtree:
                if not x.parent.tree_recursive(child_check):
                    if not x.parent.length_check(max_size):
                        continue

        # NER CHECK
        if x.leaf_recursive(reject_per):
            continue

        if x.leaf_recursive(reject_org):
            continue

        if x.leaf_recursive(reject_loc):
            continue

        # if x.leaf_recursive(reject_extra):
        #     # TODO: test if this clean is even worth it. -> results seem low quality.
        #     x.leaf_clean(reject_lst)

        # children check, TODO: remove ??
        # if x.syn_type in ccg_allowed and ccg_allowed[x.syn_type] == "NP" and x.tree_recursive(child_check) and x not in extra_subtree:
        #     print(x.get_sent())
        #     continue

        if x.syn_type in ccg_allowed.keys():
            collected.append((ccg_allowed[x.syn_type], x.syn_type, x.get_sent().strip()))
        elif x in extra_subtree:
            collected.append((extra_subtree[x], x.syn_type, x.get_sent().strip()))

    unique = sorted(list(set(collected)))
    return unique


def phrase_to_combos(
    num: int,
    listleft: list[tuple[str, str, str]],
    listright: list[tuple[str, str, str]],
    global_comb_counter: int
) -> tuple[list[tuple[int, int, str, str, str, str, str]], int]:

    combos: list[tuple[int, int, str, str, str, str, str]] = []
    # global duplicate set
    if duplicate_check:
        global duplicate_set

    # local duplicate set
    word_set_l: set[str] = set()
    if not (len(listleft) and len(listright)):
        return []

    for l_word_type, l_org, l_phrase in listleft:
        word_set_r: set[str] = set()
        for r_word_type, r_org, r_phrase in listright:
            if l_word_type == r_word_type or type_mix(l_word_type, r_word_type):
                if l_phrase == r_phrase:
                    continue
                # TODO: test for bug with removal subphrases?
                elif l_phrase in word_set_l or r_phrase in word_set_r:
                    continue
                elif l_phrase in verb_reject_lst or r_phrase in verb_reject_lst:
                    continue

                if duplicate_check:
                    # remove if encountered earlier
                    if (l_phrase, r_phrase) in duplicate_set:
                        continue
                    else:
                        duplicate_set.add((l_phrase, r_phrase))

                combos.append((global_comb_counter, num, r_word_type, l_org, r_org, l_phrase, r_phrase))
                global_comb_counter += 1
        word_set_l.add(l_phrase)
        word_set_r = set()

    return sorted(list(set(combos))), global_comb_counter


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    parser.add_argument("-d", required=False, default=False, help="disable duplicates")
    parser.add_argument("-v", required=False, default=False, help="verbose")
    args = parser.parse_args()

    dataset = args.dataset
    duplicate_check = args.d

    print_info = args.v
    if dataset == "SICK":
        # TODO, update path
        ccgfiles = glob.glob(f"datasets/{dataset}/*_ccg.pl")
    else:
        ccgfiles = glob.glob(f"datasets/{dataset}/*_cc_ccg.pl")

    print(ccgfiles)
    assert ccgfiles
    ccgfiles.sort()

    # TODO, update path
    os.makedirs(f"Results/{dataset}", exist_ok=True)

    for file in ccgfiles:
        if "fracas" in file:
            continue

        if duplicate_check:
            duplicate_set: set[tuple[str, str]] = set()

        file_name = file.rsplit(r"/", 1)[-1].replace(".pl", "")
        print(file_name)

        ccg_open = open(file, "r")
        ccg_data: list[str] = ccg_open.readlines()
        ccg_open.close()

        if dataset == "SICK":
            sen_open = open(file.replace("_ccg.pl", "_sen.pl"), "r")
        else:
            sen_open = open(file.replace("_cc_ccg.pl", "_sen.pl"), "r")

        sen_data: list[str] = sen_open.readlines()
        sen_open.close()

        # skip until first CCG line
        counter = 0
        for counter, line in enumerate(ccg_data):
            if line[:3] == "ccg":
                break

        ccg_data = ccg_data[counter:]

        all_trees: dict[int, tree] = to_tree(ccg_data)
        problem_tuple_dict: dict[int, tuple[tree, tree]] = {}
        for line in sen_data:
            line = line.rstrip()
            if line == "":
                continue
            elif line[0] == "%":
                continue

            ccg_id, problem_id = re.findall(r"(\d+)", line)[:2]

            ccg_id = int(ccg_id)
            problem_id = int(problem_id)

            if ccg_id in all_trees:
                if problem_id in problem_tuple_dict:
                    problem_tuple_dict[problem_id] = (problem_tuple_dict[problem_id], all_trees[ccg_id])
                else:
                    problem_tuple_dict[problem_id] = all_trees[ccg_id]
            else:
                if print_info:
                    print(f"CCG Num: {ccg_id}  not found in dict")

        tsvfile = open(f"Results/{dataset}/{file_name}.tsv", "w+", newline="")
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["CombID", "SenID", "merge_tag", "W1_tag", "W2_tag", "W1", "W2"])

        global_comb_counter = 1
        for i in problem_tuple_dict.keys():
            try:
                # Prem and Hypo trees
                left, right = problem_tuple_dict[i]

                # to individual phrases
                left_phrases = tree_to_phrase(left)
                right_phrases = tree_to_phrase(right)
                    
                # merge phrases
                if len(left_phrases) and len(right_phrases):  # skip if empty
                    combs, global_comb_counter = phrase_to_combos(i, left_phrases, right_phrases, global_comb_counter)

                if i == 1190:
                    # print(left_phrases)
                    # print(right_phrases)
                    for example_line in combs:
                        print(example_line[-2],"\t" ,example_line[-1])

            except TypeError:
                if print_info:
                    print(f"CCG num {i} is broken. Is tuple: {type(problem_tuple_dict[i])}")
                continue

            for comb_write in combs:
                writer.writerow(comb_write)

        print(f"total combinations: {global_comb_counter}")
        tsvfile.close()
