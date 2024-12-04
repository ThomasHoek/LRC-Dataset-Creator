import argparse
import re
from pathlib import Path
# extract data

def get_data_from_log(sent: str) -> tuple[str, str, str]:
    """
    get_data_from_log parses the data from log files to labels

    Args:
        sent (str): sentence with label info

    Returns:
        tuple[str, str, str]: two labels and the rest of the information
    """
    # remove whitespace
    sent = re.sub(" +", " ", sent)
    sent = sent.strip()

    # split based on space
    split_sent: list[str] = sent.split(" ")

    # remove comma for target, predict and solve
    split_sent[0:2] = [x.replace(",", "").strip() for x in split_sent[0:2]]

    # get info and merge rest back together
    target = split_sent[0][1:-1]
    predict = split_sent[1]
    other = " ".join(split_sent[2:])

    return target, predict, other


# get data
def file_to_int_lst(path_str: str) -> list[int]:
    return [int(x.rstrip()) for x in open(path_str, "r").readlines()]


def write_file(inp_file, prior_text: str, number_info: str, cur_line: str) -> None:
    def to_print(x: str) -> str:
        """to_print removes regex preprocessing"""
        return x.replace("|", "\n")


    inp_file.write(to_print(prior_text))
    inp_file.write(to_print(number_info))
    inp_file.write(to_print(cur_line))
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


##############################################################################
################################ Main function ###############################
if __name__ == "__main__":
    args = parse_arguments()
    result_path = args.target.split("/")[:-1]

    # open files from input
    file_info = open(args.log).readlines()
    all_info = "".join([x.replace("\n", "|") for x in file_info])

    # regex match and remove empty whitespace lines
    regex_match = r"(ERROR|Error|Inconsistency in node types \(entail\/8\)\||\d+: )"
    re_split = re.split(regex_match, all_info)
    re_split = [x for x in re_split if x.rstrip()]

    # meta_data = to_print(re_split.pop(0))
    meta_counter = 0
    for line in re_split:
        if "Length of jobs" in line:
            break
        else:
            meta_counter += 1
    re_split  = re_split[meta_counter+1:]

    # seperate results from line
    re_split[-1] = re_split[-1].split("------------------------------------------------------")[0]

    # make folders and data
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

    # set flags and size
    re_split_length = len(re_split)
    index_pointer = 0
    while index_pointer != re_split_length:
        defected_flag = False
        incost_flag = False
        aethel_flag = False
        error_flag = False
        conj_flag = False
        sv1_flag = False
        string_flag = False
        prior_text = ""

        # get first / next line
        cur_line = re_split[index_pointer]

        # error, inconsistant etc messages
        while not cur_line[:-2].isdigit():
            # print(cur_line[:-1])
            if "error" in cur_line.lower():
                error_flag = True

            # keep error data
            prior_text += cur_line

            # stay in loop until number is found
            index_pointer += 1
            cur_line = re_split[index_pointer]

        # check if number in aethel
        number_info = cur_line
        num = int(cur_line[:-2])

        # get info and paste info file
        index_pointer += 1
        cur_line = re_split[index_pointer]
        target, predict, other = get_data_from_log(cur_line)
        defected_flag = "Defected" in cur_line
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
                print(cur_line, defected_flag, target, predict)
                raise NotImplementedError("Should not reach this")

        # write all info to file
        write_file(file, prior_text, number_info, cur_line)

        if defected_flag:
            write_file(d_all, prior_text, number_info, cur_line)

        # error tracking
        if error_flag:
            write_file(error_file, prior_text, number_info, cur_line)

        index_pointer += 1
