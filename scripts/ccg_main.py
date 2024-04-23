import re
import ccg_parse
from ccg_class import tree, leaf
from typing import Callable
import csv
import os

# import iso3166
# FIXME: dont filter country NERS?


#  update to compatible list
word_list = {
    # 2.	NN	Noun, singular or mass
    r"NN": "NP",
    # 13.	NNS	Noun, plural
    r"NNS": "NP",
    # 14.	NNP	Proper noun, singular
    r"NNP": "NP",
    # 15.	NNPS	Proper noun, plural
    r"NNPS": "NP",

    # ------- Change to VP
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
np_parent_lst = [r"n", r"n/n", "n/pp", r"n\n", "n\n", "n/n"]

#  update to compatible list
ccg_allowed = {
    r"n": "NP",
    # r"np": "NP",
    # r"pp": "NP_rel",
    # r"n/pp": "SPHRASE",
    # r"np:nb": "NP",
    # r"s:dcl": "PHRASE",
    # r"s:dcl\np": "NP_rel",
    r"(s:dcl\np)/np": "NP_rel",
    # r"(s:dcl\np)\(s:dcl\np)": "PHRASE",
    # r"s:ng": "PHRASE",
    r"s:ng\np": "NP_rel",
    r"(s:ng\np)/np": "NP_rel",
    r"(s:ng\np)\(s:ng\np)": "NP_rel",
}

ccg_allowed_clean = {
    r"n": "NP",
    # r"np": "NP",
    # r"pp": "NP_rel",
    # r"n/pp": "SPHRASE",
    # r"np:nb": "NP",
    # r"s:dcl": "PHRASE",
    # r"s:dcl\np": "NP_rel",
    r"(s\np)/np": "phrasal",
    # r"(s:dcl\np)\(s:dcl\np)": "PHRASE",
    # r"s:ng": "PHRASE",
    r"s\np": "phrasal",
    r"(s\np)/np": "phrasal",
    r"(s\np)\(s\np)": "phrasal",
}

# country_long_lst: list[str] = list(iso3166.countries_by_name) + list(iso3166.countries_by_alpha2) + list(iso3166.countries_by_alpha3)


def to_tree(ccg_inp: list[str]) -> dict[int, tree]:
    """Parses list of CCG strings into a dict of trees. Dict key is CCG num."""
    ccg_data: list[list[tuple[int, str]]] = ccg_parse.parse_data(ccg_inp)
    return ccg_parse.parse_class(ccg_data)


def tree_to_phrase(tree_inp: tree, clean: bool = False,
                   plural_bool: bool = False) -> list[tuple[str, str, str] | tuple[str, str, bool, str]]:
    """
    tree_to_phrase Find every interesting phrase inside a CCG tree.

    Args:
        tree_inp (tree): CCG tree
        clean (bool, optional): To clean away :dlc, :ng etc. Defaults to False.
        plural_bool (bool, optional): extra return for plurals. Currently ommited due to minimal difference in results.

    Returns:
        list[tuple[str, str, str]: list of (merge category, specific category, sentence)
        OR if plural_bool
        list[tuple[str, str, bool, str]]: list of (merge category, specific category, plural, sentence)
    """
    global dataset
    collected: list[tuple[str, str, str] | tuple[str, str, bool, str]] = []
    reject_lst = [",", "'", "'", "\\'", r"'", r"\'", "’"]
    reject_extra: Callable[[leaf], bool] = lambda x_lamb: x_lamb.word in reject_lst

    # TODO: add countries back | add information for country synnonyms???
    reject_per: Callable[[leaf], bool] = lambda x_lamb: x_lamb.BIO_ner == "I-PER"
    reject_org: Callable[[leaf], bool] = lambda x_lamb: x_lamb.BIO_ner == "I-ORG"
    child_check: Callable[[tree], bool] = lambda x_lamb: x_lamb.syn_type not in np_parent_lst and x_lamb.combinator != "conj"

    # FIXME: config file
    max_size = 7

    # ========= LEAVES =========
    for word in tree_inp.get_leaves([]):
        if word.POS in word_list.keys():
            #  add LX check if NP
            if (word_list[word.POS] == "NP") and \
                    word.parent.syn_type in np_parent_lst and \
                    not word.parent.length_check(max_size) and \
                    not word.parent.tree_recursive(child_check):
                continue

            if plural_bool:
                collected.append((word_list[word.POS], word.POS, "NNS" in word.POS, word.word.strip()))
            else:
                collected.append((word_list[word.POS], word.POS, word.word.strip()))

    # ========= TREES =========
    for x in tree_inp.gen_subtrees():
        if x.length_check(max_size):
            continue

        # TODO: properly implement a parent check with theory. For now keep all.
        # # check if no outer
        # if x.parent_check():
        #     if x.parent.ccg in ccg_allowed:
        #         if not x.parent.parent_label_reject(child_check):
        #             if not x.parent.length_check(max_size):
        #                 continue

        if x.leaf_recursive(reject_extra):
            x.leaf_clean(reject_lst)

        if dataset == "SICK":
            # FIXME: make config file
            if x.leaf_recursive(reject_per):
                continue

            if x.leaf_recursive(reject_org):
                continue

        if x.syn_type in ccg_allowed and ccg_allowed[x.syn_type] == "NP" and x.tree_recursive(child_check):
            continue

        if clean:
            if x.syn_type_clean in ccg_allowed_clean.keys():
                if plural_bool:
                    plural = x.leaf_recursive(lambda x_lamb: x_lamb.BIO_pos == "NNS" or x_lamb.BIO_pos == "NNPS")
                    collected.append((ccg_allowed_clean[x.syn_type_clean], x.syn_type_clean, plural, x.get_sent().strip()))
                else:
                    collected.append((ccg_allowed_clean[x.syn_type_clean], x.syn_type_clean, x.get_sent().strip()))
        else:
            if x.syn_type in ccg_allowed.keys():
                if plural_bool:
                    plural = x.leaf_recursive(lambda x_lamb: x_lamb.BIO_pos == "NNS" or x_lamb.BIO_pos == "NNPS")
                    collected.append((ccg_allowed[x.syn_type], x.syn_type, plural, x.get_sent().strip()))
                else:
                    collected.append((ccg_allowed[x.syn_type], x.syn_type, x.get_sent().strip()))

    unique = list(set(collected))
    return unique


def phrase_to_combos(
    num: int,
    listleft: list[tuple[str, str, str] | tuple[str, str, bool, str]],
    listright: list[tuple[str, str, str] | tuple[str, str, bool, str]],
) -> list[tuple[int, str, str, str, str, str] | tuple[int, str, str, str, bool, bool, str, str]]:

    combos: list[tuple[int, str, str, str, str, str] | tuple[int, str, str, str, bool, bool, str, str]] = []

    if len(listleft[0]) == 3:
        for l_word_type, l_org, l_phrase in listleft:
            for r_word_type, r_org, r_phrase in listright:
                if l_word_type == r_word_type:
                    if l_phrase == r_phrase:
                        continue

                    combos.append((num, r_word_type, l_org, r_org, l_phrase, r_phrase))
                    combos.append((num, r_word_type, r_org, l_org, r_phrase, l_phrase))

    elif len(listleft[0]) == 4:
        for l_word_type, l_org, l_plural, l_phrase in listleft:
            for r_word_type, r_org, r_plural, r_phrase in listright:
                if l_word_type == r_word_type:
                    if l_phrase == r_phrase:
                        continue

                    combos.append((num, r_word_type, l_org, r_org, l_plural, r_plural, l_phrase, r_phrase))
                    combos.append((num, r_word_type, r_org, l_org, r_plural, l_plural, r_phrase, l_phrase))

    return list(set(combos))


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on",
                        choices=["SICK", "PPDB"])
    parser.add_argument("-v", required=False, default=False, help="verbose")
    args = parser.parse_args()

    dataset = args.dataset
    print_info = args.v

    if dataset == "SICK":
        ccgfiles = glob.glob(f"datasets/{dataset}/*_ccg.pl")
    else:
        ccgfiles = glob.glob(f"datasets/{dataset}/*_cc_ccg.pl")

    assert ccgfiles
    ccgfiles.sort()
    os.makedirs(f"Results/{dataset}/all_found", exist_ok=True)
    os.makedirs(f"Results/{dataset}/all_found_clean", exist_ok=True)
    for file in ccgfiles:
        if "fracas" in file:
            continue

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
        problem_tuple_dict: dict[int, tree | tuple[tree, tree]] = {}
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

        tsvfile = open(f"Results/{dataset}/all_found/{file_name}.tsv", "w+", newline="")
        tsv_clean = open(f"Results/{dataset}/all_found_clean/{file_name}.tsv", "w+", newline="")
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["SenID", "merge_tag", "W1_tag", "W2_tag", "W1_plural", "W2_plural", "W1", "W2"])

        writer_clean = csv.writer(tsv_clean, delimiter="\t", lineterminator="\n")
        writer_clean.writerow(["SenID", "merge_tag", "W1_tag", "W2_tag", "W1_plural", "W2_plural", "W1", "W2"])

        for i in problem_tuple_dict.keys():
            try:
                # Prem and Hypo trees
                left, right = problem_tuple_dict[i]

                # to individual phrases
                left_phrases = tree_to_phrase(left)
                right_phrases = tree_to_phrase(right)

                # merge phrases
                combs = phrase_to_combos(i, left_phrases, right_phrases)
            except TypeError:
                if print_info:
                    print(f"CCG num {i} is broken. Is tuple: {isinstance(problem_tuple_dict[i], tuple)}")
                continue

            for comb_write in combs:
                writer.writerow(comb_write)

            left_phrases = tree_to_phrase(left, clean=True)
            right_phrases = tree_to_phrase(right, clean=True)
            combs_clean = phrase_to_combos(i, left_phrases, right_phrases)

            for comb_write in combs_clean:
                writer_clean.writerow(comb_write)

        tsvfile.close()
        tsv_clean.close()
