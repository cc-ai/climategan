import os
from pathlib import Path
import argparse


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(*args):
    s = " ".join(args)
    print(bcolors.HEADER + "\n --- " + s + "\n" + bcolors.ENDC)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ignore", nargs="+", default=None)
    opts = parser.parse_args()

    ignores = set(opts.ignore) if opts.ignore else set()

    tests = Path(__file__).parent.glob("test_*.py")

    for test_path in tests:
        name = test_path.stem.split("test_")[1]
        if name not in ignores:
            title = ">>> RUNNING {} <<<".format(test_path.name)
            print(bcolors.OKBLUE)
            print("=" * len(title))
            print(title)
            print("=" * len(title))
            print(bcolors.ENDC)
            status = os.system("python {}".format(str(test_path)))
            if status != 0:
                error = ">>>>>>>>>> Error <<<<<<<<<<"
                print(bcolors.FAIL)
                print("=" * len(error))
                print(error)
                print("=" * len(error))
            else:
                ok = ">>> DONE {} <<<".format(test_path.name)
                print(bcolors.OKGREEN)
                print("=" * len(ok))
                print(ok)
                print("=" * len(ok))
            print(bcolors.ENDC)
            print("\n\n\n")
