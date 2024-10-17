from copy import deepcopy
import re
from dataclasses import dataclass
from numpy import add
import ccg_parse
from ccg_class import tree, leaf
from typing import Callable, Optional
import csv
import os


@dataclass(order=True)
class phrase_info:
    m_category: str  # general category
    s_category: str  # specific category
    sentence: str
    lemma: Optional[str] = None


@dataclass(order=True)
class phrase_pair:
    CombID: int
    ProbID: int
    merge_tag: str
    W1_tag: str
    W2_tag: str
    W1: str
    W2: str
    lemma_left: Optional[str] = None
    lemma_right: Optional[str] = None

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
def tree_to_phrase(tree_inp: tree, lemma_check: bool) -> list[phrase_info]:
    """
    tree_to_phrase Find every interesting phrase inside a CCG tree.

    Args:
        tree_inp (tree): CCG tree
        lemma_check (bool): Adds lemmas to the output

    Returns:
        list[tuple[str, str, str]: list of (merge category, specific category, sentence)
        or
        list[tuple[str, str, str, str]: list of (merge category, specific category, sentence, lemma)
    """
    global dataset
    collected: list[phrase_info] = []  # | tuple[str, str, bool, str]] = []
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
            word_phrase = phrase_info(word_list[word.POS], word.POS, word.word.strip())
            if lemma_check:
                word_phrase.lemma = word.lemma.strip()
            collected.append(word_phrase)

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
            add_tuple = phrase_info(ccg_allowed[x.syn_type], x.syn_type, x.get_sent().strip())
            if lemma_check:
                add_tuple.lemma = x.get_sent(lemma=True).strip()

            collected.append(add_tuple)
        elif x in extra_subtree:
            add_tuple = phrase_info(extra_subtree[x], x.syn_type, x.get_sent().strip())

            if lemma_check:
                add_tuple.lemma = x.get_sent(lemma=True).strip()
            collected.append(add_tuple)
    return collected


def phrase_to_combos(
    problem_num: int,
    listleft: list[phrase_info],
    listright: list[phrase_info],
    global_comb_counter: int
) -> tuple[list[phrase_pair], int]:

    combos: list[phrase_pair] = []
    # global duplicate set
    if duplicate_check:
        global duplicate_set

    # local duplicate set
    word_set_l: set[str] = set()
    if not (len(listleft) and len(listright)):
        return []

    # r_word_type, r_org, r_phrase
    for left_phrase in listleft:
        word_set_r: set[str] = set()
        for right_phrase in listright:
            if left_phrase.m_category == right_phrase.m_category or type_mix(left_phrase.m_category, right_phrase.m_category):
                # skip if itself
                if left_phrase.sentence == right_phrase.sentence:
                    continue
                # TODO: test for bug with removal subphrases?
                elif left_phrase.sentence in word_set_l or right_phrase.sentence in word_set_r:
                    continue
                elif left_phrase.sentence in verb_reject_lst or right_phrase.sentence in verb_reject_lst:
                    continue

                if duplicate_check:
                    # remove if encountered earlier
                    if (left_phrase.sentence, right_phrase.sentence) in duplicate_set:
                        continue
                    else:
                        duplicate_set.add((left_phrase.sentence, right_phrase.sentence))

                combo = phrase_pair(global_comb_counter, problem_num, left_phrase.m_category,
                                    left_phrase.s_category, right_phrase.s_category,
                                    left_phrase.sentence, right_phrase.sentence)
                if lemma_check:
                    combo.lemma_left = left_phrase.lemma
                    combo.lemma_right = right_phrase.lemma

                combos.append(combo)
                global_comb_counter += 1
                word_set_r.add(right_phrase.sentence)
        word_set_l.add(left_phrase.sentence)
        word_set_r = set()

    return combos, global_comb_counter


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    parser.add_argument("-d", required=False, default=False, help="disable duplicates")
    parser.add_argument("-l", required=False, default=True, help="add Lemma data")
    parser.add_argument("-v", required=False, default=False, help="verbose")
    args = parser.parse_args()

    dataset = args.dataset
    duplicate_check = args.d
    lemma_check = args.l

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
    os.makedirs(f"lex_pairs/{dataset}", exist_ok=True)

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

        tsvfile = open(f"lex_pairs/{dataset}/{file_name}.tsv", "w+", newline="")
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        if lemma_check:
            writer.writerow(["CombID", "ProbID", "merge_tag", "W1_tag", "W2_tag", "W1", "W2", "L1", "L2"])
        else:
            writer.writerow(["CombID", "ProbID", "merge_tag", "W1_tag", "W2_tag", "W1", "W2"])

        global_comb_counter = 1
        for i in problem_tuple_dict.keys():
            try:
                # Prem and Hypo trees
                left, right = problem_tuple_dict[i]

                # to individual phrases
                left_phrases: list[phrase_info] = tree_to_phrase(left, lemma_check)
                right_phrases: list[phrase_info] = tree_to_phrase(right, lemma_check)

                # merge phrases
                if len(left_phrases) and len(right_phrases):  # skip if empty
                    combs, global_comb_counter = phrase_to_combos(i, left_phrases, right_phrases, global_comb_counter)

                # if i == 1190:
                #     # print(left_phrases)
                #     # print(right_phrases)
                #     for example_line in combs:
                #         print(example_line[-2],"\t" ,example_line[-1])

            except TypeError:
                if print_info:
                    print(f"CCG num {i} is broken. Is tuple: {type(problem_tuple_dict[i])}")
                continue

            for cw in combs:
                comb_str = [cw.CombID, cw.ProbID, cw.merge_tag, cw.W1_tag, cw.W2_tag, cw.W1, cw.W2]

                if lemma_check:
                    comb_str += [cw.lemma_left, cw.lemma_right]

                writer.writerow(comb_str)

        print(f"total combinations: {global_comb_counter}")
        tsvfile.close()
