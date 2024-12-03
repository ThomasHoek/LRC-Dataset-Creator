import json
from collections import Counter

banlist = [""]


for data_set in ["train", "dev"]:
    with open(f'snli_1.0_{data_set}.jsonl', 'r') as json_file:
        json_list = list(json_file)

    counter = 0
    file_name = f'snli_{data_set}_NN.jsonl'
    file_name= file_name.replace("train", "train_5")
    with open(file_name, 'w') as f:
        for json_str in json_list:
            result = json.loads(json_str)
            if result["gold_label"] == "neutral":
                continue

            if data_set == "train" and len(result["annotator_labels"]) == 5:
                c = Counter(result["annotator_labels"])
                if c.most_common(1)[0][1] == 5:
                    f.writelines([json_str])
            elif data_set == "dev":
                f.writelines([json_str])
