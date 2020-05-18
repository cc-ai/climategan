"""File running all tests found as tests/test_*.py
"""
import argparse
import os
from pathlib import Path

import torch


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_title(text, fname, color):
    title = f">>> {text} {fname} <<<"
    print(color)
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(bcolors.ENDC)


def print_start(fname):
    print_title("RUNNING", fname, bcolors.OKBLUE)


def print_ok(fname):
    print_title("DONE", fname, bcolors.OKGREEN)


def print_error(fname):
    print_title("ERROR", fname, bcolors.FAIL)


def print_header(*args):
    """Print nice colored header
    """
    s = " ".join(args)
    print(bcolors.HEADER + "\n --- " + s + "\n" + bcolors.ENDC)


def tprint(*args):
    """Tensor Print
    """
    to_print = []
    for a in args:
        if isinstance(a, torch.Tensor):
            if len(a.shape) == 0:
                to_print.append(a.item())
            else:
                to_print.append(a.detach().numpy())
        elif isinstance(a, torch.Size):
            to_print.append(list(a))
        else:
            to_print.append(a)
    print(" ".join(map(str, to_print)))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
parser.add_argument("-i", "--ignore", nargs="+", default=None)
args = parser.parse_args()


if __name__ == "__main__":
    # Ignore tests specified via --ignore
    ignores = set(args.ignore) if args.ignore else set()

    # Find all tests
    tests = Path(__file__).parent.glob("test_*.py")

    passed = []
    failed = []

    for test_path in tests:
        # iterating over all test files
        name = test_path.stem.split("test_")[1]
        if name not in ignores:
            print_start(test_path.name)
            # ----------------------
            # -----  Run Test  -----
            # ----------------------
            status = os.system(
                "python {} --config={}".format(str(test_path), args.config)
            )

            if status != 0:
                print_error(test_path.name)
                failed.append(test_path.name)
            else:
                print_ok(test_path.name)
                passed.append(test_path.name)

            print("\n\n\n")

    passed_nb = "all"
    if failed:
        passed_nb = len(passed)
        print(bcolors.FAIL + "{} tests failed".format(len(failed)))
        for name in failed:
            print("  - {}".format(name))
        print(bcolors.ENDC)
    print(bcolors.OKGREEN + "{} tests passed".format(passed_nb) + bcolors.ENDC)
