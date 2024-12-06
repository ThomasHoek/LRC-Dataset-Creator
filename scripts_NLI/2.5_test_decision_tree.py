import re
import json
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

# full
train_dataset = "merged_LRC_words"
model_save_path = "models/NLI/tasksource_full/"

train_data = f"lex_preds/{train_dataset}/NLI/pred/inter/predictions_train.tsv"
val_data = f"lex_preds/{train_dataset}/NLI/pred/inter/predictions_validation.tsv"
# test_data = "lex_preds/merged_LRC_words/NLI/pred/inter/predictions_test.tsv"

target_names = ['disjoint', 'forwardentailment', 'independent', 'reverseentailment', 'synonym']
tree_para = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6],
    # 'class_weight': [None, 'balanced']
    }
test_result_save_path = f"{model_save_path}/testset"

with open('scripts_NLI/templates.json') as json_data:
    SNLI_templates_json: list[dict[str, str]] = json.load(json_data)
    SNLI_templates = [[line["prem"], line["hyp"]] for line in SNLI_templates_json]
    json_data.close()


def predictions_to_LOOCV(train_list):
    vars_template = 6
    for i in range(0, len(train_list[0]), vars_template):
        slice = np.delete(train_list, list(range(i, i+vars_template)), axis=1)
        yield slice

def to_slice(data, slice_idx):
    return np.delete(data, list(range(6*slice_idx, 6*(1 + slice_idx))),axis=1)

# CombID	head	tail	preds	probs
NLI_train = pd.read_csv(train_data, delimiter="\t")
train_x = [literal_eval(x) for x in NLI_train["preds"]]
test_y = list(NLI_train["label"])

temlate_scores = {}
temlate_scores2 = {}

for template_idx, train_slice in enumerate(predictions_to_LOOCV(train_list=train_x)):
    print("-"*5, "EXLUDED:", "-"*5)
    print(SNLI_templates[template_idx])
    template_str = SNLI_templates[template_idx][0] + "_" + SNLI_templates[template_idx][1]
    

    regression = DecisionTreeClassifier(random_state=0)
    clf = GridSearchCV(regression, tree_para, cv=10)
    clf.fit(train_slice, test_y)

    print(f"Best Score: {clf.best_score_}")
    print(f"Best params: {clf.best_params_}")

    temlate_scores[template_str] = clf.best_score_

    print("----Phrase dataset----")
    NLI_test_phrase = pd.read_csv("lex_preds/ppdb_phrase/NLI/pred/inter/predictions_ppdb_scrape_disjoint.tsv", delimiter="\t") 
    phrase_test_x = list(NLI_test_phrase["preds"])
    phrase_test_x = [literal_eval(x) for x in phrase_test_x]
    phrase_test_x = to_slice(phrase_test_x, template_idx)
    NLI_test_phrase["pred"] = clf.predict(phrase_test_x)
    print(classification_report(NLI_test_phrase["label"], NLI_test_phrase["pred"]))
    temlate_scores2[template_str] = accuracy_score(NLI_test_phrase["label"], NLI_test_phrase["pred"])


    print("----META----")
    dot_data_count = tree.export_graphviz(clf.best_estimator_,
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


    dot_data_count = tree.export_graphviz(clf.best_estimator_,
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
    print('all', last_template_count, node_count)
    print('-'*50)

print(dict(sorted(temlate_scores.items(), key=lambda item: item[1])))
print(dict(sorted(temlate_scores2.items(), key=lambda item: item[1])))
