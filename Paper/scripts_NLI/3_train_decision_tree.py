import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

models = glob.glob("models/NLI/*/predictions_train.tsv", recursive=True)
models_test = glob.glob("models/NLI/*/predictions_ppdb_scrape.tsv", recursive=True)

tree_para = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6],
    # 'class_weight': [None, 'balanced']
    }

for train, test in zip(models, models_test):
    model_path = train.rsplit("/", 1)[0]
    result_path = model_path.replace("models", "Results")
    print(model_path)
    NLI_train = pd.read_csv(train, delimiter="\t")

    train_x = list(NLI_train["preds"])
    train_x = [literal_eval(x) for x in train_x]
    train_y = list(NLI_train["label"])

    regression = DecisionTreeClassifier(random_state=0)
    clf = GridSearchCV(regression, tree_para, cv=10)
    clf.fit(train_x, train_y)
    print(f"Best Score: {clf.best_score_}")
    print(f"Best params: {clf.best_params_}")

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=750)
    tree.plot_tree(clf.best_estimator_, filled=True,rounded=True, ax=axes)

    os.makedirs(result_path, exist_ok=True)
    plt.savefig(f"{result_path}/tree.png")

    # =====================
    NLI_test = pd.read_csv(test, delimiter="\t")
    test_x = list(NLI_test["preds"])
    test_x = [literal_eval(x) for x in test_x]
    NLI_test["pred"] = clf.best_estimator_.predict(test_x)

    # -----
    NLI_test.to_csv(f"{result_path}/predicts_all.tsv", sep="\t", index=False)

    # -------------------------------
    os.makedirs(f"{result_path}/all", exist_ok=True)
    os.makedirs(f"{result_path}/org", exist_ok=True)
    os.makedirs(f"{result_path}/add", exist_ok=True)

    cr = classification_report(NLI_test["label"], NLI_test["pred"])
    cr_file = open(f"{result_path}/all/classification_all.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(NLI_test["label"], NLI_test["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"{result_path}/all/ConfusionMatrix_all.jpg")

    # -------------------------------
    org_df = NLI_test[NLI_test["meta"] == "org"]
    cr = classification_report(org_df["label"], org_df["pred"])
    cr_file = open(f"{result_path}/org/classification_org.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(org_df["label"], org_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"{result_path}/org/ConfusionMatrix_org.jpg")

    # -------------------------------
    add_df = NLI_test[NLI_test["meta"] == "add"]
    cr = classification_report(add_df["label"], add_df["pred"])
    cr_file = open(f"{result_path}/add/classification_add.txt", "w+")
    cr_file.write(cr)
    cr_file.close()

    disp = ConfusionMatrixDisplay.from_predictions(add_df["label"], add_df["pred"])
    disp.plot(xticks_rotation=45).figure_.savefig(f"{result_path}/add/ConfusionMatrix_add.jpg")

    print("========================")
