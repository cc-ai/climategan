import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import save_segmap_tensors

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_json",
    type=str,
    required=True,
    help="path to the json file where to find original data",
)
parser.add_argument(
    "--path_dir",
    type=str,
    required=True,
    help="path to the directory where to save the tensors as tensor_name.pt",
)
parser.add_argument(
    "-d", "--domain", type=str, required=True, help="domain of the images (r or s)"
)
args = parser.parse_args()


if __name__ == "__main__":
    save_segmap_tensors(args.path_json, args.path_dir, args.domain)
