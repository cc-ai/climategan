import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.utils import make_json_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--tasks",
    nargs="+",
    required=True,
    help="the list of image type like 'x', 'm', 'd', etc.",
)
parser.add_argument(
    "-a",
    "--addresses",
    nargs="+",
    required=True,
    help="the list of the corresponding data path of each task",
)
parser.add_argument(
    "-j",
    "--json_names",
    nargs="+",
    required=True,
    help="the names for the val and train json files, train being first",
)
parser.add_argument(
    "-p",
    "--val_p",
    type=float,
    required=True,
    help="pourcentage of files to go in validation set (between 0 and 1",
)
args = parser.parse_args()


if __name__ == "__main__":
    make_json_file(
        tasks=args.tasks,
        addresses=args.addresses,
        json_names=args.json_names,
        pourcentage_val=args.val_p,
    )
