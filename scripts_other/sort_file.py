import argparse
from itertools import groupby
import re


def sort_file(path: str) -> None:
    def find_id(inp_raw: list[str]) -> int:
        return int(re.search(r"\d+:", "".join(inp_raw)).group()[:-1])

    file_read = open(path)
    file_info = file_read.readlines()
    file_read.close()

    file_group: list[list[str]] = [
        list(group)
        for key, group in groupby(file_info, lambda x: x == " \n" or x == "\n")
        if not key
    ]

    dict_problem: dict[int, list[str]] = {}
    for i in file_group:
        dict_problem[find_id(i)] = i

    write_file = open(path, "w+")
    for key in sorted(dict_problem):
        write_file.write(f"{''.join(dict_problem[key])}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract annotations from Alpino xml trees"
    )

    # Arguments covering directories and files
    parser.add_argument("target", metavar="DIR", help="path to analysis folder")

    # pre-processing arguments
    args = parser.parse_args()
    return args

##############################################################################
################################ Main function ################################
if __name__ == "__main__":
    import glob

    args = parse_arguments()
    all_file_paths: list[str] = glob.glob(f'{args.target}/*.txt')
    for x in all_file_paths:
        sort_file(x)
