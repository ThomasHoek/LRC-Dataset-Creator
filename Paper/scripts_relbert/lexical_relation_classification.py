import os
import logging
from itertools import product, chain
from multiprocessing import Pool
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier
from datasets import load_dataset, load_from_disk
from nltk.tokenize import word_tokenize

# from edited_LM import RelBERT
from relbert import RelBERT
from util import fix_seed

class RelationClassification:
    def __init__(
        self,
        dataset,
        label_dict,
        target_relation=None,
        default_config: bool = False,
        config=None,
        cur_dir=""
    ):
        self.dataset = dataset
        self.label_dict = label_dict
        self.target_relation = target_relation
        self.cur_dir = cur_dir
        
        if default_config:
            self.configs = [{"random_state": 0}]
        elif config is not None:
            self.configs = [config]
        else:
            learning_rate_init = [0.001, 0.0001, 0.00001]
            hidden_layer_sizes = [100, 150, 200]
            self.configs = [
                {
                    "random_state": 0,
                    "learning_rate_init": i[0],
                    "hidden_layer_sizes": i[1],
                }
                for i in list(product(learning_rate_init, hidden_layer_sizes))
            ]

    def run_test(self, clf, x, y):
        """run evaluation on valid or test set"""
        y_pred = clf.predict(x)
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
            y, y_pred, average="macro"
        )
        p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(
            y, y_pred, average="weighted"
        )
        accuracy = sum([a == b for a, b in zip(y, y_pred.tolist())]) / len(y_pred)
        tmp = {
            "accuracy": accuracy,
            "f1_macro": f_mac,
            "f1_micro": f_mic,
            "p_macro": p_mac,
            "p_micro": p_mic,
            "r_macro": r_mac,
            "r_micro": r_mic,
            # "prediction": y_pred,
            # "true_label": y
        }
        for _l in self.label_dict:
            p, r, f, _ = precision_recall_fscore_support(
                y, y_pred, labels=[self.label_dict[_l]]
            )
            tmp[f"f1/{_l}"] = f[0]
            tmp[f"p/{_l}"] = p[0]
            tmp[f"r/{_l}"] = r[0]
        return tmp

    @property
    def config_indices(self):
        return list(range(len(self.configs)))

    def __call__(self, config_id):
        config = self.configs[config_id]
        # train
        x, y = self.dataset["train"]
        clf = MLPClassifier(**config).fit(x, y)
        report = {"classifier_config": clf.get_params()}
        # test

        x, y = self.dataset["test"]
        tmp = self.run_test(clf, x, y)
        tmp = {f"test/{k}": v for k, v in tmp.items()}
        report.update(tmp)

        if "val" in self.dataset:
            x, y = self.dataset["val"]
            tmp = self.run_test(clf, x, y)
            tmp = {f"val/{k}": v for k, v in tmp.items()}
            report.update(tmp)

        if True:
            import pickle

            # update pickle locs
            global pickle_name

            os.makedirs(f"models/relbert/{pickle_name}/", exist_ok=True)
            # FIXME save path
            with open(f"models/relbert/{pickle_name}/model.pkl", "wb") as f:
                pickle.dump(obj=clf, file=f)
            with open(f"models/relbert/{pickle_name}/model_dict.pkl", "wb") as f:
                pickle.dump(self.label_dict, file=f)

        test_x, test_y = self.dataset["test"]

        test_pred = clf.predict(test_x)
        return report, (test_y, test_pred)


def evaluate_classification(
    relbert_ckpt: str = None,
    max_length: int = 64,
    batch_size: int = 64,
    target_relation=None,
    random_seed: int = 0,
    config=None,
    validation_metric: str = "f1_micro",
    cur_dir: str = ""):
    fix_seed(random_seed)
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, "model is not trained"

    # FIXME: edited
    data_names = ["shwarz_all"]

    result = {}
    for data_name in data_names:
        # if data_name == "shwarz_all":
        #     # FIXME test path
        #     data = load_from_disk(f"{cur_dir}/../shwarz_all.hf")
        #     # data = load_from_disk(f"{cur_dir}/../big.hf")
        # else:
        #     data = load_dataset("relbert/lexical_relation_classification", data_name)

        file_dict = {
            "train": "datasets/merged_LRC_words/train.tsv",
            "test": "datasets/merged_LRC_words/test.tsv",
            "val": "datasets/merged_LRC_words/validation.tsv"
            }


        data = load_dataset('csv',
                            data_files=file_dict,
                            delimiter='\t',
                            column_names=["head", "tail", "relation", "meta"],
                            )
        print(data)
        logging.info(f"train model with {relbert_ckpt} on {data_name}")
        relations = sorted(
            list(set(list(chain(*[data[_k]["relation"] for _k in data.keys()]))))
        )
        label_dict = {r: n for n, r in enumerate(relations)}
        dataset = {}

        print("Embedding start")
        for _k in data.keys():
            _v = data[_k]
            label = [label_dict[i] for i in _v["relation"]]

            # s1 = [" ".join(word_tokenize(x)) for x in _v["s1"]]
            # s2 = [" ".join(word_tokenize(x)) for x in _v["s2"]]

            # x_tuple = [tuple(_x) for _x in zip(_v["head"], _v["tail"], s1, s2)]
            # x_back = [tuple(_x) for _x in zip(_v["tail"], _v["head"], s2, s1)]

            x_tuple = [tuple(_x) for _x in zip(_v["head"], _v["tail"])]
            x_back = [tuple(_x) for _x in zip(_v["tail"], _v["head"])]

            # x_tuple = [tuple(_x) for _x in zip(_v["head"], _v["tail"])]
            # FIXME: null -> Null in test file.
            # FIXME:if Relbert used, keep in mind
            for x,y in x_tuple:
                if x == None or y== None:
                    print(x, y)

            x = model.get_embedding(x_tuple, batch_size=batch_size)
            x_back = model.get_embedding(x_back, batch_size=batch_size)

            x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
            dataset[_k] = [x, label]

        print("Embedding done")
        logging.info("run with default config")

        evaluator = RelationClassification(
            dataset,
            label_dict,
            target_relation=target_relation,
            default_config=True,
            cur_dir=cur_dir
        )
        metric = evaluator(0)

        # else:
        #     # can't save model?
        #     logging.info("run grid search")
        #     pool = Pool()
        #     evaluator = RelationClassification(
        #         dataset, label_dict,
        #         target_relation=target_relation, cur_dir=cur_dir
        #     )
        #     metrics = pool.map(evaluator, evaluator.config_indices)
        #     pool.close()
        #     metric = sorted(
        #         metrics, key=lambda m: m[0][f"val/{validation_metric}"], reverse=True
        #     )[0]

        metric, (y, y_pred) = metric
        result[f"lexical_relation_classification/{data_name}"] = metric
    del model
    return result, (y, y_pred)


if __name__ == "__main__":

    def thomas_custom_prompter(word_pair: tuple[str, str, str, str], template: str, mask_token: str = None):
        """Transform word pair into string prompt."""
        import re

        # hard code it in, fix later
        # template = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj> is the <mask> of <subj>"
        # template = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj> is the <mask> of <subj>. With the context <obj> in <s1> and <subj> in <s2>"
        # template = "With the context <obj> in <s1> and <subj> in <s2>. I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj> is the <mask> of <subj>."
        # template = "<obj> in <s1> and <subj> in <s2>. But I just read in the encyclopedia that <obj> is the <mask> of <subj>. <obj> in <s1> and <subj> in <s2>"

        template = "' <W1> ' <SEP> ' <W2> ' <SEP> ' <S1> ' <SEP> ' <S2> '"
        # template = "' <S1> ' <SEP> ' <S2> ' <SEP> ' <W1> ' <SEP> ' <W2> '"
        # template = "' <W1> ' <SEP> ' <W2> ' <SEP> ' <S1> ' <SEP> ' <S2> ' <SEP> ' <W1> ' <SEP> ' <W2> '"

        pickle_dir = "w1w2s1s2"
        os.makedirs()
        
        token_mask = "<mask>"
        token_subject = "<subj>"
        token_object = "<obj>"
        # assert len(word_pair) == 2, word_pair
        subj = word_pair[0]
        obj = word_pair[1]
        s1 = word_pair[2]
        s2 = word_pair[3]

        assert (
            token_subject not in subj
            and token_object not in subj
            and token_mask not in subj
        )
        assert (
            token_subject not in obj
            and token_object not in obj
            and token_mask not in obj
        )

        sentence = re.sub("<subj>", subj, template)
        sentence = re.sub("<obj>", obj, sentence)
        sentence = re.sub("<s1>", s1, sentence)
        sentence = re.sub("<s2>", s2, sentence)

        if mask_token is not None:
           sentence = sentence.replace(token_mask, mask_token)
        # print(sentence)
        # exit()
        return sentence

    # relbert.lm.custom_prompter = thomas_custom_prompter

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    from time import gmtime, strftime
    import pandas as pd
    from sklearn.metrics import classification_report

    if True:
        import warnings

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    pickle_dir = "pickle_saved"
    pickle_names = ["relbert-roberta-base", "relbert-roberta-large"]

    for pickle_name in pickle_names:
        result, (y, y_pred) = evaluate_classification(
            f"relbert/{pickle_name}",
            cur_dir=cur_dir, random_seed=0,
            config="default_config"
        )


        df = pd.DataFrame.from_dict(result)

        curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        df.to_csv(f"models/relbert/{pickle_name}/{curr_time}_test_thomas.tsv", sep="\t")

        y = pd.Series(y)
        y_pred = pd.Series(y_pred)
        total = pd.DataFrame(dict(y=y, y_pred=y_pred))
        print(classification_report(y, y_pred, digits=3))
        total.to_csv(f"models/relbert/{pickle_name}/{curr_time}_test_thomas_res.tsv", sep="\t")
