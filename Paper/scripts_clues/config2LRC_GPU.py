# mypy: ignore-errors
# type: ignore

import numpy as np
import pandas as pd
import re
import os
import torch
from torch import nn
from random import randint
import argparse
from datetime import datetime
import logging
from transformers import set_seed
import yaml
from sklearn.metrics import (
    top_k_accuracy_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import entropy
from typing import Any
from datasets import load_metric, load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import glob

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def find_model(
    yaml_dict: dict[Any, Any], mod_inp_name: str
) -> tuple[str, dict[Any, Any]]:
    assert mod_inp_name in [x for xs in yaml_dict["models"] for x in xs.keys()]
    for model in yaml_dict["models"]:
        for model_name, model_config in model.items():
            if model_name == mod_inp_name:
                return model_name, model_config

    raise NotImplementedError("Should not reach this point")


def verb_row(row: pd.DataFrame, template_t: tuple[str, str], tokenizer, verb_dict=None):
    """
    Create a verbalization of a row (a pair of words and
    a relation label) following a template that can contains
    <W1>, <W2>, <LABEL> and <SEP> to substitute source,
    target words, the relation label and the special token SEP
    of a tokenizer. If verb_dict is not None, verb_dict is a
    dictionary that must contains pairs (key, value)
    where key is a relation label, and value is the verbalization
    of the relation label uses to sustitute <LABEL> in the template.

    Args:
      row -- a series with 'source', 'target' and 'rel'
      template_t -- a  tuple of template and context with (possible) <W1>, <W2>, <S1>, <S2>, <LABEL> and <SEP>
      tokenizer -- a tokenizer with its special tokens
      verb_dict -- dictionary with the verbalizations (values) of
        the relation labels (keys)

    Returns:
      a dictionary, {'verb':verbalization}, with the key 'verb'
      and the verbalization of the row following the template.
    """
    template: str
    w1 = str(row["source"])
    w2 = str(row["target"])

    if "sen1" in row:
        s1 = str(row["sen1"])
        s2 = str(row["sen2"])


    if ("sen1" not in row) or (s1 == "" or s2 == ""):
        template = template_t[0]
    else:
        template = template_t[1]

    lab = str(row["rel"]).lower()

    sentence = re.sub("<W1>", w1, template)
    sentence = re.sub("<W2>", w2, sentence)
    if "sen1" in row:
        sentence = re.sub("<S1>", s1, sentence)
        sentence = re.sub("<S2>", s2, sentence)
        
    sentence = re.sub("<SEP>", tokenizer.sep_token, sentence)

    if verb_dict is not None:
        if lab in verb_dict:
            lab = verb_dict[lab].strip()
        sentence = re.sub("<LABEL>", lab, sentence)

    return {"verb": sentence}


def preprocess_function(rows, tokenizer):
    """tokenize the column 'verb' of the rows"""
    inputs = tokenizer(
        rows["verb"], truncation=True, padding="max_length", max_length=64
    )
    return inputs


def results_row(row, tokenizer):
    pred = row["pred_label"]
    gold = row["real_label"]
    if pred == gold:
        row["results"] = True
    else:
        row["results"] = False

    toks_s = tokenizer.tokenize(" " + row["source"])
    toks_t = tokenizer.tokenize(" " + row["target"])
    row["toks_source"] = str(toks_s)
    row["toks_target"] = str(toks_t)
    row["n_toks_source"] = len(toks_s)
    row["n_toks_target"] = len(toks_t)
    return row


def run_model(yaml_dict: dict[Any, Any], config_name: str, model_name_inp: str):
    datadir: str = root_dir + yaml_dict["datadir"]
    modeldir: str = root_dir + yaml_dict["modeldir"]
    total_repetitions = yaml_dict["total_repetitions"]
    batch_size = yaml_dict["batch_size"]
    warm_up = yaml_dict["warm_up"]
    total_epochs = yaml_dict["total_epochs"]

    name_dataset = config_name
    yaml_config = yaml_dict[config_name]

    dict_of_rel_verb = yaml_config["dict"]
    dataset_name = yaml_config["dataset"]
    train_template = yaml_config["train_template"]
    test_template = yaml_config["test_template"]
    train_template_c = yaml_config["train_template_c"]
    test_template_c = yaml_config["test_template_c"]

    train_file = datadir + yaml_config["train_file"]
    test_file = datadir + yaml_config["test_file"]
    val_file = datadir + yaml_config["val_file"]
    # params = yaml_config["params"]

    date = datetime.now().strftime("%D-%H:%M:%S")
    if model_name_inp == "all":
        for model in yaml_config["models"]:
            for model_name, _ in model.items():
                # FIXME thomas: remove this
                if "relbert" in model_name:
                    pass
                else:
                    run_model(yaml_dict, config_name, model_name)
        return
    else:
        model_name, model_config = find_model(yaml_config, model_name_inp)

    output = f"{modeldir}/{model_name}-{config_name}"
    model_path = model_config["path"]
    # CODE FROM LRC GITHUB. SLIGHTLY MODIFIED FOR EFFICIENCY

    # global exist
    # if not exist:
    #     if os.path.exists(f"{output}"):
    #         return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exc_message = "Train templates and test templates must be lists of equal size.\nTrain template list contains {:d} templates and test template list contains {:d}"
    if len(train_template) is not len(test_template):
        raise Exception(exc_message.format(len(train_template), len(test_template)))

    if len(train_template_c) is not len(test_template_c):
        raise Exception(exc_message.format(len(train_template_c), len(test_template_c)))

    # tokenizer for model
    print("-"*20)
    print(model_path)
    print("-"*20)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    # find the type of tokenizer
    m = (
        "bert"
        if tokenizer.tokenize("a")[0] == tokenizer.tokenize(" a")[0]
        else "roberta"
    )
    if m == "roberta":
        dict_of_rel_verb = {
            x_k: " {}".format(x_v) for (x_k, x_v) in dict_of_rel_verb.items()
        }

    # create output dir, if it does not exist
    os.makedirs(output, exist_ok=True)
    os.makedirs(f"{output}/scores", exist_ok=True)

    msgFinetuning = """Starting fine-tuning with:
    - model: {:s}
    - train file: {:s} 
    - test file: {:s}
    - val file: {:s}
    - train templates: {:s}
    - test templates: {:s}
    *****************************************"""
    logging.info(
        msgFinetuning.format(
            model_name,
            train_file,
            test_file,
            val_file if val_file is not None else "None",
            str(train_template),
            str(test_template),
        )
    )

    # PREPARE DATA
    # load train/test files to datasets dict. Also load val file, if it exists
    # datasets contains lines with three strings: source_word, target_word, rel_label
    data_files = {"train": train_file, "test": test_file}
    if val_file is not None:
        data_files["val"] = val_file

    # laugh	rack	disjoint	cog | RANDOM
    col_name = ["source", "target", "labels", "meta"]

    all_data = load_dataset(
        path="csv",
        data_files=data_files,
        sep="\t",
        header=None,
        column_names=col_name,
        keep_default_na=False,
    )
    all_data = all_data.remove_columns(["meta"])
    print(all_data["train"]["source"][:5])
    print(all_data["train"]["target"][:5])
    print(all_data["train"]["labels"][:5])

    # create the column 'rel', copy of column 'labels', if exists.
    all_data = all_data.map(lambda x: {"rel": x["labels"]})

    # trasform column 'labels' to a integer with a label id. Needed for the tokenizer
    all_data = all_data.class_encode_column("labels")

    # print(all_data["train"][0])
    # Calculate number of synsets of the words in test dataset
    print("Calculating number synsets for words in test dataset....")
    source_words = np.unique(np.array(all_data["test"]["source"]))
    target_words = np.unique(np.array(all_data["test"]["target"]))

    all_words = np.unique(np.concatenate([source_words, target_words]))
    synsets_dict = {}
    for word in all_words:
        synsets_dict[word] = len(wn.synsets(word))

    # load metric
    metric_name = "f1"
    metric = load_metric(metric_name, trust_remote_code=True)

    def compute_metrics(eval_pred):
        """
        Compute metrics for a Trainer.

        Args:
        eval_pred: object of type transformers.EvalPrediction. It is a tuple with
        predictions (logits) and real labels.

        Returns:
        A dictionary of metrics {'name_metric1':value1,...}
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    # original way
    # seeds to avoid equal trainings
    if False:
        seeds = [randint(1, 100) for n in range(total_repetitions)]
        while len(set(seeds)) is not total_repetitions:
            seeds = [randint(1, 100) for n in range(total_repetitions)]
    else:  # modified way such that seed are always the same
        seeds = list(range(1, total_repetitions + 1))
        assert len(set(seeds)) == total_repetitions

    # should be one, so we ignore
    # for i in range(total_repetitions):
    # print("****** Repetition: " + str(i + 1) + "/" + str(total_repetitions))
    set_seed(seeds[0])
    # seed is always 1
    NUM_LABELS: int = all_data["train"].features["labels"].num_classes
    NAME_LABELS: list[str] = all_data["train"].features["labels"].names

    id2lab: dict[int, str] = dict()
    for i_lab in range(NUM_LABELS):
        id2lab[i_lab] = NAME_LABELS[i_lab]

    lab2id: dict[str, int] = {v: k for k, v in id2lab.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS,
        label2id=lab2id, id2label=id2lab,
        ignore_mismatched_sizes=True)

    model = model.to(device)

    # verbalize the datasets with template
    all_data["train"] = all_data["train"].map(
        verb_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "template_t": (train_template, train_template_c),
            "verb_dict": dict_of_rel_verb,
        },
    )
    all_data["test"] = all_data["test"].map(
        verb_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "template_t": (test_template, test_template_c),
            "verb_dict": dict_of_rel_verb,
        },
    )
    if val_file != None:
        all_data["val"] = all_data["val"].map(
            verb_row,
            fn_kwargs={
                "tokenizer": tokenizer,
                "template_t": (test_template, test_template_c),
                "verb_dict": dict_of_rel_verb,
            },
        )

    # encode data for language model
    encoded_all_data = all_data.map(
        preprocess_function,
        batched=True,
        batch_size=None,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # separate the splits in datasets dict
    encoded_verb_train = encoded_all_data["train"]
    if val_file != None:
        encoded_verb_val = encoded_all_data["val"]
    encoded_verb_test = encoded_all_data["test"]

    encoded_verb_train.set_format("torch")
    if val_file != None:
        encoded_verb_val.set_format("torch")
    encoded_verb_test.set_format("torch")


    args_train = TrainingArguments(
        output_dir="models/clues/my_checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="epoch" if val_file != None else "no",
        save_strategy="epoch" if val_file != None else "no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        optim="adamw_torch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=warm_up,
        # fp16=True,
        logging_steps=10,
        load_best_model_at_end=True if val_file != None else False,
        metric_for_best_model=metric_name,
        num_train_epochs=total_epochs,
        report_to="all",
        seed=seeds[0],
        save_total_limit=3,  # patience if val_file != None else 0,
    )

    trainer = Trainer(
        model,  # model to train
        args_train,  # arguments to train
        train_dataset=encoded_verb_train,
        eval_dataset=encoded_verb_val if val_file != None else None,
        tokenizer=tokenizer,  # it is needed the tokenizer that encoded the data for batch
        compute_metrics=compute_metrics,  # to compute metric of the model,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if val_file != None and patience != None else None
    )
    #  'rel': Value(dtype='string', id=None), 
    #  'target': Value(dtype='string', id=None),
    # {'source': Value(dtype='string', id=None),
    #  'verb': Value(dtype='string', id=None),

    # ignored: rel, target, source, verb.
    #  'labels': ClassLabel(names=['antonym', 'attribute', 'contains', 'coordinated', 'event', 'hyper', 'material', 'mero', 'random', 'synonym'],id=None),
    #  'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
    #  'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None)}


    # start training
    trainer.train()
    trainer.save_model(f"{output}")

    # predict test
    predicciones = trainer.predict(test_dataset=encoded_verb_test)
    # calculate the predicted labels 0/1 based on the field predictions of the object predicciones
    # predicciones.predictions contains the logits
    pred = np.argmax(predicciones.predictions, axis=1)
    print(
        metric.compute(
            predictions=pred, references=predicciones.label_ids, average="micro"
        )
    )

    real_rel_test = encoded_verb_test.features["labels"].int2str(
        encoded_verb_test["labels"]
    )
    pred_rel_test = encoded_verb_test.features["labels"].int2str(pred)
    results_acc = classification_report(
        real_rel_test, pred_rel_test, digits=4, output_dict=True
    )
    print(results_acc)
    encoded_verb_test.set_format("numpy")
    results_words = pd.DataFrame(
        {
            "pred_label": pred,
            "pred_rel": pred_rel_test,
            "real_label": predicciones.label_ids,
            "real_rel": real_rel_test,
            "source": encoded_verb_test["source"],
            "target": encoded_verb_test["target"],
        }
    )
    results_words = results_words.apply(results_row, axis=1, tokenizer=tokenizer)

    sfmax = nn.Softmax(dim=1)
    probs = sfmax(torch.tensor(predicciones.predictions))
    probs_df = pd.DataFrame(
        probs.numpy(), columns=encoded_verb_test.features["labels"].names
    )
    chaos = entropy(probs, axis=1, base=2)
    chaos_df = pd.DataFrame(chaos, columns=["entropy"])
    nsynsets = results_words.apply(
        lambda x: [synsets_dict[x["source"]], synsets_dict[x["target"]]],
        axis=1,
        result_type="expand",
    )
    nsynsets.columns = ["nsynsests_source", "nsynsests_target"]

    results_words = pd.concat([results_words, probs_df, chaos_df, nsynsets], axis=1)

    now = datetime.now()
    now = now.strftime("%y-%m-%d_%H-%M-%S")
    fname = output + "/scores/" + name_dataset + "_I" + "_" + now
    with open((fname + ".txt"), "w") as f:
        print(vars(args), file=f)
        print(date, file=f)
        print(results_acc, file=f)

        results_words.to_csv(fname + ".csv", index=False)

    with open(f"{output}/sents.txt", "w+") as sents:
        sents.write(train_template)
        sents.write("\n")
        sents.write(test_template)
        sents.write("\n")
        sents.write(train_template_c)
        sents.write("\n")
        sents.write(test_template_c)

    extension = "csv"
    result = glob.glob(f"{output}/scores/*.{extension}")
    result.sort(key=os.path.getctime, reverse=True)
    print(result)

    data = pd.read_csv(result[0])
    print(classification_report(data["real_rel"], data["pred_rel"], digits=3))
    print("\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="huggingface model from config or all"
    )
    parser.add_argument("--name", required=True, help="name from config to be used")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
    cur_dir: str = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(f"{cur_dir}/config.yaml") as stream:
            yaml_dict = yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found")
    except yaml.YAMLError as exc:
        raise NotImplementedError(f"Error in parsing file, {exc}")

    # debug
    exist = int(yaml_dict["overwrite"]) > 0
    print(exist)

    if args.name == "all":
        for i in yaml_dict:
            if i not in ["datadir", "modeldir", "total_repetitions", "batch_size", "warm_up", "total_epochs", "overwrite"]:            
                print(i, args.model)
                run_model(yaml_dict, i, args.model)
    elif args.name in yaml_dict:
        print(args.name, args.model)
        run_model(yaml_dict, args.name, args.model)
    else:
        raise NotImplementedError("Name not found in config")
