def to_template(SNLI_templates: list[tuple[str, str]], ID: str, scrapeID: str, w1: str, w2: str, label: str, meta: str):
    for templatenum, (prem, hyp) in enumerate(SNLI_templates):
        prem_1 = prem.replace("NP", w1)
        prem_2 = prem.replace("NP", w2)

        hyp_1 = hyp.replace("NP", w1)
        hyp_2 = hyp.replace("NP", w2)

        yield f"{ID}\t{scrapeID}\t{templatenum}\t{prem_1}\t{hyp_2}\t{label}\t{meta}\n"
        yield f"{ID}\t{scrapeID}\t{templatenum}\t{prem_2}\t{hyp_1}\t{label}\t{meta}\n"


def to_template_no_nums(SNLI_templates: list[tuple[str, str]], ID: str, w1: str, w2: str, label: str):
    for templatenum, (prem, hyp) in enumerate(SNLI_templates):
        prem_1 = prem.replace("NP", w1)
        prem_2 = prem.replace("NP", w2)

        hyp_1 = hyp.replace("NP", w1)
        hyp_2 = hyp.replace("NP", w2)
        yield f"{ID}\t{templatenum}\t{prem_1}\t{hyp_2}\t{label}\n"
        yield f"{ID}\t{templatenum}\t{prem_2}\t{hyp_1}\t{label}\n"


if __name__ == "__main__":
    import argparse
    import csv
    import json
    import glob
    import os

    dir_path = str(os.path.dirname(os.path.realpath(__file__)))

    with open(f'{dir_path}/templates.json') as json_data:
        SNLI_templates_json: list[dict[str, str]] = json.load(json_data)
        json_data.close()
    SNLI_templates = [(line["prem"], line["hyp"]) for line in SNLI_templates_json]

    
    datafile = open("datasets/ppdb_phrase/ppdb_scrape_disjoint.tsv", "r+")

    os.makedirs("Results/NLI_manual", exist_ok=True)
    insertfile = open("Results/NLI_manual/templates_insert_ppdb_scrape.tsv", "w+")
    insertfile.write(f"CombID\tSenID\ttemplatenum\tprem\thyp\tlabel\tmeta\n")

    # ID	scrapeID	w1	w2	label	meta
    for csvline in csv.DictReader(datafile, delimiter="\t"):
        counter = 0
        for template_line in to_template(SNLI_templates, csvline["ID"], csvline["scrapeID"],
                                         csvline["w1"], csvline["w2"], csvline["label"], csvline["meta"]):
            insertfile.write(template_line)

    # ==============================================================================
    for data_part in ["train", "test"]:
        datafile = open(f"datasets/merged_LRC_words/{data_part}.tsv", "r+")

        os.makedirs("Results/NLI_manual", exist_ok=True)
        insertfile = open(f"Results/NLI_manual/templates_insert_{data_part}.tsv", "w+")
        insertfile.write("ID\ttemplatenum\tprem\thyp\tlabel\n")

        # ID	scrapeID	w1	w2	label	meta
        total_lines = 0
        for counter, csvline in enumerate(csv.DictReader(datafile, delimiter="\t", fieldnames=["w1", "w2", "label", "meta"])):
            for template_line in to_template_no_nums(SNLI_templates, counter, csvline["w1"], csvline["w2"], csvline["label"]):
                total_lines += 1
                insertfile.write(template_line)

            if data_part == "train":
                if total_lines > 100_000:
                    break
            elif data_part == "test":
                if total_lines > 10_000:
                    break
