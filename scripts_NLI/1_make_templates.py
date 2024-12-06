def to_template(SNLI_templates: list[tuple[str, str]], ID: str, scrapeID: str, w1: str, w2: str):
    for templatenum, (prem, hyp) in enumerate(SNLI_templates):
        prem_1 = prem.replace("NP", w1)
        prem_2 = prem.replace("NP", w2)

        hyp_1 = hyp.replace("NP", w1)
        hyp_2 = hyp.replace("NP", w2)
        yield f"{ID}\t{scrapeID}\t{templatenum}\t{w1}\t{w2}\t{prem_1}\t{hyp_2}\n"
        yield f"{ID}\t{scrapeID}\t{templatenum}\t{w1}\t{w2}\t{prem_2}\t{hyp_1}\n"


def get_results(file_path: str, part: str):
    datafile = open(file_path, "r+")

    os.makedirs(f"{dir_path}/../lex_preds/{dataset}/NLI/templates/", exist_ok=True)
    insertfile = open(f"{dir_path}/../lex_preds/{dataset}/NLI/templates/{part}.tsv", "w+")

    insertfile.write(f"CombID\tProbID\ttemplatenum\thead\ttail\tprem\thyp\n")

    total_counter = 0
    for csvline in csv.DictReader(datafile, delimiter="\t"):
        counter = 0
        for template_line in to_template(SNLI_templates, csvline["CombID"], csvline["ProbID"], csvline["W1"], csvline["W2"]):
            insertfile.write(template_line)
            counter += 1
            total_counter += 1

        if total_counter > limit:
            break


def to_template_baseline(SNLI_templates: list[tuple[str, str]], ID: str, w1: str, w2: str, label: str):
    for templatenum, (prem, hyp) in enumerate(SNLI_templates):
        prem_1 = prem.replace("NP", w1)
        prem_2 = prem.replace("NP", w2)

        hyp_1 = hyp.replace("NP", w1)
        hyp_2 = hyp.replace("NP", w2)
        yield f"{ID}\t{templatenum}\t{w1}\t{w2}\t{prem_1}\t{hyp_2}\t{label}\n"
        yield f"{ID}\t{templatenum}\t{w1}\t{w2}\t{prem_2}\t{hyp_1}\t{label}\n"


def get_results_baseline(file_path: str, part: str):
    datafile = open(file_path, "r+")

    os.makedirs(f"{dir_path}/../lex_preds/{dataset}/NLI/templates", exist_ok=True)
    insertfile = open(f"{dir_path}/../lex_preds/{dataset}/NLI/templates/{part}.tsv", "w+")

    insertfile.write(f"CombID\ttemplatenum\thead\ttail\tprem\thyp\tlabel\n")

    counter = 0
    if part == "ppdb_scrape_disjoint":
        for csvline in csv.DictReader(datafile, delimiter="\t"):
            for template_line in to_template_baseline(SNLI_templates, counter, csvline["w1"], csvline["w2"], csvline['label']):
                insertfile.write(template_line)
            counter += 1
    else:
        for csvline in [x.rstrip() for x in datafile.readlines()]:
            # laugh	rack	disjoint	cog | RANDOM
            w1, w2,label, _  = csvline.split("\t")
            for template_line in to_template_baseline(SNLI_templates, counter, w1, w2, label):
                insertfile.write(template_line)
            counter += 1
            # if counter > 5000:
            #     break


if __name__ == "__main__":
    import argparse
    import csv
    import json
    import glob
    import os
    import numpy as np

    parser = argparse.ArgumentParser(description="File used to create templates from lexical pairs.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    parser.add_argument("--part", required=True, metavar="FILES", help="Part of dataset to test on")
    parser.add_argument("--limit", required=False, metavar="FILES", help="Set an int limit")
    parser.add_argument("--baseline", required=False, default=False, metavar="FILES", help="boolean flag if the dataset has a different format, eg baselines")
    args = parser.parse_args()
    dataset = args.dataset
    part = args.part
    limit: float | None = args.limit
    baseline: bool = args.baseline

    # global var
    if limit is None:
        limit = np.inf
    else:
        limit = int(limit)

    dir_path = str(os.path.dirname(os.path.realpath(__file__)))
    with open(f'{dir_path}/templates.json') as json_data:
        SNLI_templates_json: list[dict[str, str]] = json.load(json_data)
        json_data.close()
    SNLI_templates = [(line["prem"], line["hyp"]) for line in SNLI_templates_json]

    print(f"{dir_path}/../lex_pairs/{dataset}/")
    if part == "all":
        files_found = glob.glob(f"{dir_path}/../lex_pairs/{dataset}/*.tsv")
        print(files_found)
        for i in files_found:
            part_strip = i.split("/")[-1].replace("_ccg.tsv", "").replace(f"{dataset}_", "")
            get_results(i, part_strip)

    else:
        try:
            files_found = glob.glob(f"{dir_path}/../lex_pairs/{dataset}/{dataset}_{part}_ccg.tsv")
            get_results(files_found[0], part)
        except IndexError:
            if baseline and len(files_found) == 0:
                files_found = glob.glob(f"datasets_original/{dataset}/{part}.tsv")
                print(files_found)
                assert len(files_found) == 1
                get_results_baseline(files_found[0], part)
            elif len(files_found) == 0:
                files_found = glob.glob(f"lex_pairs/{dataset}/{part}_ccg.tsv")
                print(files_found)
                assert len(files_found) == 1
                get_results(files_found[0], part)
                
            else:
                raise IndexError