import argparse
import os
from collections import Counter
from pathlib import Path

import comet_ml
import yaml
from addict import Dict
from comet_ml import config


def parse_tags(tags_str):
    all_tags = set(t.strip() for t in tags_str.split(","))
    keep_tags = set()
    remove_tags = set()
    for t in all_tags:
        if "!" in t or "~" in t:
            remove_tags.add(t[1:])
        else:
            keep_tags.add(t)
    return all_tags, keep_tags, remove_tags


def select_lambdas(vars):
    """
    Create a specific file with the  painter's lambdas

    Args:
        vars (dict): output of locals()
    """
    opts = vars["opts"]
    dev = vars["args"].dev
    lambdas = opts.train.lambdas.G.p
    if not dev:
        with open("./painter_lambdas.yaml", "w") as f:
            yaml.safe_dump(lambdas.to_dict(), f)


def parse_value(v: str):
    """
    Parses a string into bool or list or int or float or returns it as is

    Args:
        v (str): value to parse

    Returns:
        any: parsed value
    """
    if v.lower() == "false":
        return False
    if v.lower() == "true":
        return True
    if v.startswith("[") and v.endswith("]"):
        return [
            parse_value(sub_v)
            for sub_v in v.replace("[", "").replace("]", "").split(", ")
        ]
    if "." in v:
        try:
            vv = float(v)
            return vv
        except ValueError:
            return v
    else:
        try:
            vv = int(v)
            return vv
        except ValueError:
            return v


def parse_opts(summary):
    """
    Parses a flatten_opts summary into an addict.Dict

    Args:
        summary (list(dict)): List of dicts from exp.get_parameters_summary()

    Returns:
        addict.Dict: parsed exp params
    """
    opts = Dict()
    for item in summary:
        k, v = item["name"], parse_value(item["valueCurrent"])
        if "." in k:
            d = opts
            for subkey in k.split(".")[:-1]:
                d = d[subkey]
            d[k.split(".")[-1]] = v
        else:
            opts[k] = v
    return opts


def has_right_tags(exp: comet_ml.Experiment, keep: set, remove: set) -> bool:
    """
    All the "keep" tags should be in the experiment's tags
    None of the "remove" tags should be in the experiment's tags.

    Args:
        exp (comet_ml.Experiment): experiment to select (or not)
        keep (set): tags the exp should have
        remove (set): tags the exp cannot have

    Returns:
        bool: should this exp be selected
    """
    tags = set(exp.get_tags())
    has_all_keep = keep.intersection(tags) == keep
    has_any_remove = remove.intersection(tags)
    return has_all_keep and not has_any_remove


if __name__ == "__main__":
    # ------------------------
    # -----  Parse args  -----
    # ------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_id", type=str, default="")
    parser.add_argument(
        "-d",
        "--download_dir",
        type=str,
        default=None,
        help="Where to download the images",
    )
    parser.add_argument(
        "-s", "--step", default="last", type=str, help="`last`, `all` or `int`"
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        default="./",
        type=str,
        help="if download_dir is not specified, download into base_dir/exp_id[:8]/",
    )
    parser.add_argument(
        "-t",
        "--tags",
        default="",
        type=str,
        help="download all images of all with a set of tags",
    )
    parser.add_argument(
        "-i",
        "--id_length",
        default=8,
        type=int,
        help="Length of the experiment's ID substring to make dirs: exp.id[:id_length]",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="dry run: no mkdir, no download",
    )
    parser.add_argument(
        "-p",
        "--post_processings",
        default="",
        type=str,
        help="comma separated string list of post processing functions to apply",
    )
    parser.add_argument(
        "-r",
        "--running",
        default=False,
        action="store_true",
        help="only select running exps",
    )
    args = parser.parse_args()
    print(args)

    # -------------------------------------
    # -----  Create post processings  -----
    # -------------------------------------

    POST_PROCESSINGS = {"select_lambdas": select_lambdas}
    post_processes = list(
        filter(
            lambda p: p is not None,
            [POST_PROCESSINGS.get(k.strip()) for k in args.post_processings.split(",")],
        )
    )

    # ------------------------------------------------------
    # -----  Create Download Dir from download_dir or  -----
    # -----  base_dir/exp_id[:args.id_length]          -----
    # ------------------------------------------------------

    download_dir = Path(args.download_dir or Path(args.base_dir)).resolve()
    if not args.dev:
        download_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # -----  Check step  -----
    # ------------------------

    step = None
    try:
        step = int(args.step)
    except ValueError:
        step = args.step
        assert step in {"last", "all"}

    api = comet_ml.api.API()

    # ---------------------------------------
    # -----  Select exps based on tags  -----
    # ---------------------------------------
    if not args.tags:
        assert args.exp_id
        exps = [api.get_experiment_by_id(args.exp_id)]
    else:
        all_tags, keep_tags, remove_tags = parse_tags(args.tags)
        download_dir = download_dir / "&".join(sorted(all_tags))

        print("Selecting experiments with tags", all_tags)
        conf = dict(config.get_config())
        exps = api.get_experiments(
            workspace=conf.get("comet.workspace"),
            project_name=conf.get("comet.project_name") or "climategan",
        )
        exps = list(filter(lambda e: has_right_tags(e, keep_tags, remove_tags), exps))
        if args.running:
            exps = [e for e in exps if e.alive]

    # -------------------------
    # -----  Print setup  -----
    # -------------------------

    print(
        "Processing {} experiments in {} with post processes {}".format(
            len(exps), str(download_dir), post_processes
        )
    )
    assert all(
        [v == 1 for v in Counter([e.id[: args.id_length] for e in exps]).values()]
    ), "Experiment ID conflict, use a larger --id_length"

    for e, exp in enumerate(exps):
        # ----------------------------------------------
        # -----  Setup Current Download Directory  -----
        # ----------------------------------------------
        cropped_id = exp.id[: args.id_length]
        ddir = (download_dir / cropped_id).resolve()
        if not args.dev:
            ddir.mkdir(parents=True, exist_ok=True)

        # ------------------------------
        # -----  Fetch image list  -----
        # ------------------------------
        ims = [asset for asset in exp.get_asset_list() if asset["image"] is True]

        # -----------------------------------
        # -----  Filter images by step  -----
        # -----------------------------------

        if step == "last":
            curr_step = max(i["step"] or -1 for i in ims)
            if curr_step == -1:
                curr_step = None
        else:
            curr_step = step

        ims = [i for i in ims if (i["step"] == curr_step) or (step == "all")]

        ddir = ddir / str(curr_step)
        if not args.dev:
            ddir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------
        # -----  Store experiment's link and opts  -----
        # ----------------------------------------------
        summary = exp.get_parameters_summary()
        opts = parse_opts(summary)
        if not args.dev:
            with open("./url.txt", "w") as f:
                f.write(exp.url)
            with open("./opts.yaml", "w") as f:
                yaml.safe_dump(opts.to_dict(), f)

        # ------------------------------------------
        # -----  Download png files with curl  -----
        # ------------------------------------------
        print(
            " >>> Downloading exp {}'s image at step `{}` into {}".format(
                cropped_id, args.step, str(ddir)
            )
        )

        for i, im in enumerate(ims):
            if not Path(im["fileName"] + "_{}.png".format(curr_step)).exists():
                print(
                    "\nDownloading exp {}/{} image {}/{}: {} in {}".format(
                        e + 1, len(exps), i + 1, len(ims), im["fileName"], ddir
                    )
                )
                if not args.dev:
                    assert len(im["curlDownload"].split(" > ")) == 2
                    curl_command = im["curlDownload"].split(" > ")[0]
                    file_stem = Path(im["curlDownload"].split(" > ")[1]).stem

                    file_path = (
                        f'"{str(ddir / file_stem)}_{cropped_id}_{curr_step}.png"'
                    )

                    signal = os.system(f"{curl_command} > {file_path}")
        for p in post_processes:
            p(locals())
