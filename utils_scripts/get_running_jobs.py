import os
import subprocess
from pathlib import Path
import yaml
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--search_dir", default="/miniscratch/schmidtv/vicc/omnigan/runs/painter"
    )
    parser.add_argument("-r", "--running", default=True, action="store_true")
    parser.add_argument(
        "-i", "--ids", nargs="+", type=str, default=[], help="List of ids to look for",
    )
    args = parser.parse_args()
    search_dir = args.search_dir

    sq = subprocess.run(
        f"squeue -u {os.environ['USER']}".split(), stdout=subprocess.PIPE
    ).stdout.decode("utf-8")

    jobs = [j.strip() for j in sq.split("\n")[1:] if j.strip()]

    running_ids = []
    if args.running:
        running_ids = [j.split()[0] for j in jobs if j.split()[4] == "R"]

    all_ids = args.ids + running_ids

    print("Searching in", search_dir)

    paths = sorted(Path(search_dir).iterdir(), key=os.path.getmtime, reverse=True)

    for p in paths:
        yam = p / "opts.yaml"
        if yam.exists():
            with yam.open("r") as f:
                opt = yaml.safe_load(f)
                p_id = str(opt.get("jobID"))
            if p_id in all_ids:
                print(opt.get("jobID"), str(p))
                all_ids.remove(p_id)
        if not all_ids:
            break

    if all_ids:
        print("Could not find matchin opts for", *all_ids)
        for jid in all_ids:
            print([j for j in jobs if j.split()[0] == jid][0])
