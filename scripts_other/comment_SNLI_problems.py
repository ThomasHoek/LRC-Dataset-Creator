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

sen_file = f"datasets/SNLI_5_anno/snli_1.0_train_5_anno_sen.pl"
sen_data = open(sen_file, "r").readlines()

ccg_file = f"datasets/SNLI_5_anno/snli_1.0_train_5_anno_ccg.pl"
ccg_data = open(ccg_file, "r").readlines()


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

bad_sents = ['A subway station where numerous people are standing and one woman is sitting and reading .',
             'On a rainy day , a biker in biking gear grins and holds his bike above his head triumphantly while British-looking cars drive by on a wet road to the right .',
             'A couple in a town where tourist are visiting but they are unhappy .',
             'A lot of people are in a room where some are sitting and some are standing .']

bad_numbers = []
wrong_list = []
for prob_id, tup in problem_tuple_dict.items():

    if type(tup) != tuple:
        bad_numbers.append(prob_id)
        continue

    (tree_A, tree_B) = tup
    # if tree_A.syn_type != tree_B.syn_type:
    # if tree_A.syn_type[0] != tree_B.syn_type[0]:
        # wrong_list.append((tree_A.syn_type[0], tree_B.syn_type[0]))
        # bad_numbers.append(prob_id)
    if tree_A.get_sent('') in bad_sents:
        bad_numbers.append(prob_id)
    elif tree_B.get_sent('') in bad_sents:
        bad_numbers.append(prob_id)

print(f"BAD PROBLEMS: {len(bad_numbers)} |  {Counter(bad_numbers)}")

sen_file_fix = f"datasets/SNLI_5_anno/snli_1.0_train_5_anno_sen_fix.pl"
sen_data_fix = open(sen_file_fix, "w+")

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

            
        # else:
            # print(f"Skipping: {num}")

    if print_flag:
        sen_data_fix.write(line)

    if counter > 7500:
        break
