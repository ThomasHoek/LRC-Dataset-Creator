import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

train_data = "models/NLI/tasksource_full/inter/predictions_templates_insert_train_full.tsv"
test_data = "models/NLI/tasksource_full/inter/predictions_templates_insert_test_full.tsv"

tree_para = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6],
    # 'class_weight': [None, 'balanced']
    }
model_save_path = "models/NLI/tasksource_full/"
test_result_save_path = "models/NLI/tasksource_full/testset"


NLI_train = pd.read_csv(train_data, delimiter="\t")

train_x = list(NLI_train["preds"])
train_x = [literal_eval(x) for x in train_x]
train_y = list(NLI_train["label"])

regression = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(regression, tree_para, cv=10)
clf.fit(train_x, train_y)
print(f"Best Score: {clf.best_score_}")
print(f"Best params: {clf.best_params_}")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=750)
tree.plot_tree(clf.best_estimator_, filled=True, rounded=True, ax=axes, class_names=['disjoint', 'forwardentailment', 'independent', 'reverseentailment', 'synonym'])

os.makedirs(model_save_path, exist_ok=True)
plt.savefig(f"{model_save_path}/tree.png")

with open(f"{model_save_path}/model.pkl", 'wb') as f:
    pickle.dump(clf,f)

# =====================
NLI_test = pd.read_csv(test_data, delimiter="\t")
test_x = list(NLI_test["preds"])
test_x = [literal_eval(x) for x in test_x]
NLI_test["pred"] = clf.best_estimator_.predict(test_x)

os.makedirs(test_result_save_path, exist_ok=True)
NLI_test.to_csv(f"{test_result_save_path}/predicts_all.tsv", sep="\t", index=False)

cr = classification_report(NLI_test["label"], NLI_test["pred"])
cr_file = open(f"{test_result_save_path}/classification_all.txt", "w+")
cr_file.write(cr)
cr_file.close()

disp = ConfusionMatrixDisplay.from_predictions(NLI_test["label"], NLI_test["pred"])
disp.plot(xticks_rotation=45).figure_.savefig(f"{test_result_save_path}/ConfusionMatrix_all.jpg")