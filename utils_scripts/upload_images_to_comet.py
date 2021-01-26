import comet_ml  # noqa: F401
from pathlib import Path
import sys
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).resolve().parent.parent))

from omnigan.utils import upload_images_to_exp

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--images_path", type=str, default=".")
    parser.add_argument("-p", "--project_name", type=str, default="omnigan-eval")
    args = parser.parse_args()

    exp = upload_images_to_exp(
        Path(args.images_path).resolve(),
        exp=None,
        project_name=args.project_name,
        sleep=0.25,
    )

    exp.end()
