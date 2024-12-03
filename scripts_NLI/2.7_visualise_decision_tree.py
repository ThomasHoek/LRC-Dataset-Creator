import re
import os
import json
import copy
import random
import pickle
import pydotplus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


dataset = "merged_LRC_words"
data_name = "tasksource_full"
test_data = f"lex_preds/{dataset}/NLI/pred/inter/predictions_test.tsv"
test_info = f"datasets_original/{dataset}/test.tsv"

model_dir = f"models/NLI/{data_name}/model.pkl"

with open('scripts_NLI/templates.json') as json_data:
    SNLI_templates_json: list[dict[str, str]] = json.load(json_data)
    json_data.close()
SNLI_templates = [[line["prem"], line["hyp"]] for line in SNLI_templates_json]


with open(model_dir, 'rb') as f:
    clf = pickle.load(f)
    clf= clf.best_estimator_

# =====================
NLI_test = pd.read_csv(test_data, delimiter="\t")
NLI_test_info = pd.read_csv(test_info, delimiter="\t", names=["head", "tail", "label", "meta"])
test_x_str = list(NLI_test["preds"])
test_x = [literal_eval(x) for x in test_x_str]

newline = "<BR/>"
feature_names = []
for template_i in range(0, len(SNLI_templates)):
    for direction in [0, 1]:
        for label_nli in ["Entail", "Contradict", "Neutral"]:
            local_templates = copy.deepcopy(SNLI_templates[template_i])
            if direction:
                dir = ("tail", "head")
            else:
                dir = ("head", "tail")

            local_templates[0] = local_templates[0].replace("NP", f"&lt;{dir[0]}&gt;")
            local_templates[1] = local_templates[1].replace("NP", f"&lt;{dir[1]}&gt;")
            feature_names.append(f"{f'{newline}'.join(local_templates)}{newline}{label_nli}")

            # feature_names.append(f"Template {template_i} Direction {direction} {label_nli}")

target_names = ['disjoint', 'forwardentailment', 'independent', 'reverseentailment', 'synonym']

dot_data_small = tree.export_graphviz(clf,
                                # out_file='models/NLI/tasksource_full/tree_full.png',
                                max_depth=3,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")
graph_small = pydotplus.graph_from_dot_data(dot_data_small)
graph_small.write_pdf(f'models/NLI/{data_name}/tree_part.pdf')

dot_data_full: str = tree.export_graphviz(clf,
                                # out_file='models/NLI/tasksource_full/tree_full.png',
                                # max_depth=4,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")


di_phrase = 'digraph Tree {node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;'
di_phrase_replace = 'digraph Tree {graph [nodesep=0.1]; node [width=0.1, margin=0.1, shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;'
dot_data_full = dot_data_full.replace(f'{di_phrase}',
                                      f'{di_phrase_replace} ranksep = 2 ; ')
with open(f"models/NLI/{data_name}/tree_full2.txt", "w") as text_file:
    text_file.write(dot_data_full)
    
graph_full = pydotplus.graph_from_dot_data(dot_data_full)
graph_full.write_pdf(f'models/NLI/{data_name}/tree_full2.pdf')


# ====================================================
# COUNT TEST
dot_data_count = tree.export_graphviz(clf,
                                class_names=target_names,
                                max_depth=3,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")
graph_count = pydotplus.graph_from_dot_data(dot_data_count)
# print(graph_count)


node_count = 0
last_template_count = {}
for node in graph_count.get_node_list():
    if "label" not in node.get_attributes():
        continue

    feature_num = re.findall(r"<SUB>([0-9]*)</SUB>", node.get_attributes()['label'])
    if feature_num:
        node_count += 1
        if int(feature_num[0]) // 6 not in last_template_count:
            last_template_count[int(feature_num[0]) // 6] = 0

        last_template_count[int(feature_num[0]) // 6] += 1
last_template_count = dict(sorted(last_template_count.items()))
print('3', last_template_count, node_count)


dot_data_count = tree.export_graphviz(clf,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")
graph_count = pydotplus.graph_from_dot_data(dot_data_count)
# print(graph_count)


node_count = 0
last_template_count = {}
for node in graph_count.get_node_list():
    if "label" not in node.get_attributes():
        continue

    feature_num = re.findall(r"<SUB>([0-9]*)</SUB>", node.get_attributes()['label'])
    if feature_num:
        node_count += 1
        if int(feature_num[0]) // 6 not in last_template_count:
            last_template_count[int(feature_num[0]) // 6] = 0

        last_template_count[int(feature_num[0]) // 6] += 1
last_template_count = dict(sorted(last_template_count.items()))
print('max', last_template_count, node_count)

# ====================================================

# SLOW AND CONVOLUTED WAY TO SET FEATURE NAMES MYSELF
depth_setting = 3
dot_data: str = tree.export_graphviz(clf,
                                # out_file='models/NLI/tasksource_full/tree_full.png',
                                max_depth=depth_setting,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")
graph = pydotplus.graph_from_dot_data(dot_data)

dot_data = tree.export_graphviz(clf,
                                # out_file='models/NLI/tasksource_full/tree_full.png',
                                max_depth=depth_setting,
                                # feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True).replace("\n", "")
graph_no_feature = pydotplus.graph_from_dot_data(dot_data)

# empty all nodes, i.e.set color to white and number of samples to zero
for node in graph.get_node_list():
    if node.get_attributes().get('label') is None:
        continue
    if 'samples = ' in node.get_attributes()['label']:
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = 0'
        node.set('label', '<br/>'.join(labels))
        node.set_fillcolor('white')


# ADD words in only FOR green
# idx = 16119
# print(idx)

test_x = list(NLI_test["preds"])
test_x = [literal_eval(x) for x in test_x]
NLI_test["pred"] = clf.predict(test_x)
print(classification_report(NLI_test["label"], NLI_test["pred"]))

# ppdb_scrape_disjoint
print("----------------------------------------")
print("PHRASE")
NLI_test_phrase = pd.read_csv("lex_preds/ppdb_phrase/NLI/pred/inter/predictions_ppdb_scrape_disjoint.tsv", delimiter="\t") 
test_x = list(NLI_test_phrase["preds"])
test_x = [literal_eval(x) for x in test_x]
NLI_test_phrase["pred"] = clf.predict(test_x)
print(classification_report(NLI_test_phrase["label"], NLI_test_phrase["pred"]))

print("----------------------------------------")

idx = random.randint(0, len(test_x) - 1)
head_tail_info = (NLI_test_info.iloc[idx]["head"], NLI_test_info.iloc[idx]["tail"])
print(head_tail_info)
print(NLI_test["label"].iloc[idx])
samples = [list(test_x[idx])]
print(clf.predict(samples))

decision_paths = clf.decision_path(samples)

for decision_path in decision_paths:
    for n, node_value in enumerate(iterable=decision_path.toarray()[0]):
        if node_value == 0:
            continue
        try:
            node_no_feature = graph_no_feature.get_node(str(n))[0]
        except IndexError:
            # hotfix for smaller trees
            continue
        node = graph.get_node(str(n))[0]

        feature_num = re.findall(r"<SUB>([0-9]*)</SUB>", node_no_feature.get_attributes()['label'])

        node.get_attributes()['label'] = node.get_attributes()['label'].replace("&lt;head&gt", f"{head_tail_info[0]}")
        node.get_attributes()['label'] = node.get_attributes()['label'].replace("&lt;tail&gt", f"{head_tail_info[1]}")

        if feature_num:
            feature_num = int(feature_num[0])

            for nli_label in ["Entail", "Contradict", "Neutral"]:
                node.get_attributes()['label'] = node.get_attributes()['label'].replace(f"{nli_label}", f"{nli_label} [{round(test_x[idx][feature_num], 3)}]")

        node.set_fillcolor('green')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))

graph.write_pdf(f'models/NLI/{data_name}/tree_single.pdf')
