import re
from collections import Counter
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root) + "/scripts_ccg")

import ccg_parse
from ccg_class import tree, leaf



def to_tree(ccg_inp: list[str]) -> dict[int, tree]:
    """Parses list of CCG strings into a dict of trees. Dict key is CCG num."""
    ccg_data: list[list[tuple[int, str]]] = ccg_parse.parse_data(ccg_inp)
    return ccg_parse.parse_class(ccg_data)


def get_problem_dict(ccg_data, sen_data):
    # skip until first CCG line
    counter = 0
    for counter, line in enumerate(ccg_data):
        if line[:3] == "ccg":
            break


    ccg_data = ccg_data[counter:]
    all_trees: dict[int, tree] = to_tree(ccg_data)

    problem_tuple_dict: dict[str, tuple[tree, tree]] = {}
    for line in sen_data:
        line = line.rstrip()
        if line == "":
            continue
        elif line[0] == "%":
            continue

        # FIXME: IMPORTANT difference between STR and INT
        ccg_id, problem_id = re.findall(r"(\d+), (.*?),", line)[0]

        ccg_id = int(ccg_id)
        # print(problem_id)

        if ccg_id in all_trees:
            if problem_id in problem_tuple_dict:
                problem_tuple_dict[problem_id] = (problem_tuple_dict[problem_id], all_trees[ccg_id])
            else:
                problem_tuple_dict[problem_id] = all_trees[ccg_id]
        else:
            pass
            # print(f"CCG Num: {ccg_id}  not found in dict")
    return problem_tuple_dict


bad_sents = ['a subway station where numerous people are standing and one woman is sitting and reading .',
             'on a rainy day , a biker in biking gear grins and holds his bike above his head triumphantly while british-looking cars drive by on a wet road to the right .',
             'a couple in a town where tourist are visiting but they are unhappy .',
             'a lot of people are in a room where some are sitting and some are standing .',
             'two hikers crossing a snowy field , with mountainous terrain behind them .']


for dataset_name in ["train", "dev", "test"]:
    str_replace = dataset_name.replace("train", "train_5")
    str_replace = str_replace + "_NN"
    sen_file = f"datasets_ccg/SNLI_NN/snli_{str_replace}_sen.pl"
    sen_data = open(sen_file, "r").readlines()

    ccg_file = f"datasets_ccg/SNLI_NN/snli_{str_replace}_ccg.pl"
    ccg_data = open(ccg_file, "r").readlines()


    problem_tuple_dict = get_problem_dict(ccg_data, sen_data)
    bad_numbers = []
    for prob_id, tup in problem_tuple_dict.items():
        if type(tup) != tuple:
            if tup.get_sent('') in bad_sents:
                bad_numbers.append(prob_id)
            continue
        
        for sent_tup in tup:
            if sent_tup.get_sent('') in bad_sents:
                bad_numbers.append(prob_id)
            continue

    sen_file_fix = f"datasets_ccg/SNLI_NN/snli_{str_replace}_sen_fix.pl"
    sen_data_fix = open(sen_file_fix, "w+")
    print(dataset_name, bad_numbers)
    counter = 0
    print_flag = False
    for line in sen_data:
        # print(line)
        if line[0] == "%":
            print_flag = False

            num = re.findall(r"(\d+)", line)[0]
            if num not in bad_numbers:
                print_flag = True
                counter += 1

        if print_flag:
            sen_data_fix.write(line)

        if dataset_name == "train" and counter > 7500:
            break
