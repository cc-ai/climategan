"""File to run an experiment:
    * loads the experiment file
    * writes the data files for each run in the experiment
        * hash.txt => git hash code
        * comet_url.txt => link to comet experiment
        * config.yaml => trainer configuration
        * exp_i.yaml => copy of the experiment file (i is the index in the run list)
    * starts the experiments with sbatch
"""
from omnigan.utils import load_exp, write_run_template, get_increased_path, write_hash
from argparse import ArgumentParser
from pathlib import Path
import yaml
import subprocess
from shutil import copyfile


def parsed_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="shared/experiment/exp_defaults.yaml",
        help="Path to the experiment to run",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="shared/experiment/template.sh",
        help="Path to sbatch template file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parsed_args()

    assert Path(args.experiment).exists()
    assert Path(args.template).exists()

    xopts = load_exp(args.experiment)

    for i in range(len(xopts.runs)):
        # setup run path
        p = get_increased_path(xopts.runs[i].trainer.output_path)
        xopts.runs[i].trainer.output_path = str(p)
        p.mkdir(parents=True)

        # write data files
        with open(p / "config.yaml", "w") as f:
            yaml.safe_dump(xopts.runs[i].trainer.to_dict(), f)
        copyfile(args.experiment, p / f"exp_{i}.yaml")
        write_run_template(xopts, i, args.template, p / "exp.sh")
        write_hash(p / "hash.txt")

        # launch sbatch job
        if not xopts.experiment.dev_mode:
            print(subprocess.check_output(f"sbatch { str(p / 'exp.sh')}", shell=True))
        print("In", str(p / "exp.sh"), "\n")
