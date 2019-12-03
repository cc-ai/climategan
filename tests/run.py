import os
from pathlib import Path
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ignore", nargs="+", default=None)
    opts = parser.parse_args()

    ignores = set(opts.ignore) if opts.ignore else set()

    tests = Path(__file__).parent.glob("test_*.py")

    for test_path in tests:
        name = test_path.stem.split("test_")[1]
        if name not in ignores:
            print(">>> RUNNING {}".format(test_path.name))
            os.system("python {}".format(str(test_path)))
            print("\n\n Done.\n\n")
