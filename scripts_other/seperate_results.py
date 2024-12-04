import argparse
import re
from pathlib import Path


def write_file(inp_file, text: str, KB_info: str | None) -> None:
    def to_print(x: str) -> str:
        """to_print removes regex preprocessing"""
        return x.strip().replace("|", "\n")

    if KB_info:
        inp_file.write(to_print(KB_info))

    inp_file.write(to_print(text))
    inp_file.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse .log files into seperate files")

    # Arguments covering directories and files
    parser.add_argument("log", metavar="log", help="path to log file")

    parser.add_argument("target", metavar="target DIR", help="target directory")
    # pre-processing arguments
    args = parser.parse_args()
    return args


KB_dict: dict[int, str] = {}
info_dict: dict[int, str] = {}
other_data: list[str] = ["Lemma Lexic", "a:DT", "gen_quant_tt", "Inconsistency in node types", "Warning:"]
if __name__ == "__main__":
    args = parse_arguments()
    result_path = args.target.split("/")[:-1]

    # open files from input
    file_info: list[str] = open(args.log).readlines()
    all_info: list[str] = [x.replace("\n", "|") for x in file_info]

    # remove start
    meta_counter = 0
    for line in all_info:
        if "Length of jobs" in line:
            break
        else:
            meta_counter += 1
    all_info = all_info[meta_counter+1:]

    # remove end
    last_dash_num = 0
    count_down = 10
    all_info.reverse()
    for idx, line in enumerate(all_info):
        if line.count('-') > 10:
            last_dash_num = idx + 1
            count_down = 10
        if count_down > 0:
            count_down -= 1
        else:
            break

    all_info = all_info[last_dash_num:]
    all_info.reverse()

    for line in all_info:
        # use KB match dict
        if match := re.search(r"(\d+) KB:", line, re.IGNORECASE):
            KB_dict[int(match.group(1))] = "KB:" + line.split("KB:")[1]
            continue
        else:
            bad_flag = False
            for bad_word in other_data:
                if re.search(bad_word, line):
                    bad_flag = True
                    break
            if bad_flag:
                continue
        if match := re.search(r"(\d+): ", line, re.IGNORECASE):
            cur_i = int(match.group(1))
            info_dict[cur_i] = line
        else:
            info_dict[cur_i] += line
        # print("GOOD", repr(line[:25]))

    Path(f"{args.target}").mkdir(parents=True, exist_ok=True)
    u_u = open(f"{args.target}/unknown_unknown.txt", "w+")
    u_y = open(f"{args.target}/unknown_yes.txt", "w+")
    u_n = open(f"{args.target}/unknown_no.txt", "w+")

    y_u = open(f"{args.target}/yes_unknown.txt", "w+")
    y_y = open(f"{args.target}/yes_yes.txt", "w+")
    y_n = open(f"{args.target}/yes_no.txt", "w+")

    n_u = open(f"{args.target}/no_unknown.txt", "w+")
    n_y = open(f"{args.target}/no_yes.txt", "w+")
    n_n = open(f"{args.target}/no_no.txt", "w+")

    d_all = open(f"{args.target}/defected_all.txt", "w+")

    error_file = open(f"{args.target}/error.txt", "w+")

    for idx, current_line in info_dict.items():
        # set flags and size
        target, predict = re.search("\[(yes|no|unknown)\], *(yes|no|unknown)", current_line).groups()
        defected_flag = "Defected" in current_line
        # match to find correct file to write to
        match (target, predict):
            case ("unknown", "unknown"):
                file = u_u

            case ("unknown", "yes"):
                file = u_y

            case ("unknown", "no"):
                file = u_n

            case ("yes", "unknown"):
                file = y_u

            case ("yes", "yes"):
                file = y_y

            case ("yes", "no"):
                file = y_n

            case ("no", "unknown"):
                file = n_u

            case ("no", "yes"):
                file = n_y

            case ("no", "no"):
                file = n_n

            case _:
                print(current_line, defected_flag, target, predict)
                raise NotImplementedError("Should not reach this")

        kb_sent = KB_dict[idx] if idx in KB_dict else None
        # write all info to file       
        write_file(file, current_line, kb_sent)

        if defected_flag:
            write_file(file, current_line, kb_sent)