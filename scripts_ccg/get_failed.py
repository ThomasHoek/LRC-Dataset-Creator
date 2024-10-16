if __name__ == "__main__":
    import os
    import glob
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Part used to create the context from. Train, Test or Trial.")
    parser.add_argument("--dataset", required=True, metavar="FILES", help="Dataset to test on")
    args = parser.parse_args()

    dataset = args.dataset

    spl_files = glob.glob(f"datasets/{dataset}/*.spl")
    output_files = [x.replace(".spl", "_output.txt") for x in spl_files]
    wrong_files = [x.replace(".spl", "_wrong.txt") for x in spl_files]
    print(spl_files, output_files)

    for out, spl, wrong in zip(output_files, spl_files, wrong_files):
        with open(out, "r") as f_out, open(spl, "r") as f_spl, open(wrong, "w+") as f_wrong:
            failed_num: list[int] = []
            for line in f_out:
                nums = re.findall("(\d*) failed", line)
                if len(nums):
                    failed_num.append(int(nums[0]))

            for counter, line in enumerate(f_spl, start=1):
                if counter in failed_num:
                    f_wrong.write(str(counter) + "\t")
                    f_wrong.write(line)
