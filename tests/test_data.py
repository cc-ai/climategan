import argparse
import sys
from pathlib import Path

from addict import Dict

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import (
    OmniListDataset,
    get_all_loaders,
    get_loader,
    get_simclr_loaders,
)
from omnigan.utils import load_test_opts
from omnigan.tutils import transforms_string


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    loaders = get_all_loaders(opts)

    ds = OmniListDataset("train", "r", opts)

    # --------------------------------
    # -----  Test task matching  -----
    # --------------------------------
    tasks = set(opts.tasks)
    tasks.add("x")
    for sample_path in ds.samples_paths:
        ds_vars = set(sample_path.keys())
        assert ds_vars.issubset(tasks)

    # --------------------------------
    # -----  Test SimCLR loaders -----
    # --------------------------------
    print("--Test simclr_loaders--")
    sim_loaders = get_simclr_loaders(opts)
    batch = Dict(next(iter(sim_loaders["train"]["r"])))
    for k, value in batch["data"].items():
        for task, tensor in value.items():
            print(
                task,
                tensor.shape,
                tensor.dtype,
                tensor.min().item(),
                tensor.max().item(),
                batch["domain"],
                batch["mode"],
            )
    print()

    # ------------------------------------
    # -----  Test transforms_string  -----
    # ------------------------------------
    print(transforms_string(loaders["train"]["r"].dataset.transform))

    batch = Dict(next(iter(loaders["train"]["r"])))
    print("Batch: ", "items, ", " ".join(batch.keys()), "keys")

    # -------------------------------
    # -----  Test batch values  -----
    # -------------------------------
    for k in batch["data"]:
        print(
            k,
            batch["data"][k].shape,
            batch["data"][k].dtype,
            batch["data"][k].min().item(),
            batch["data"][k].max().item(),
            [Path(p).name for p in batch["paths"][k]],
            batch["domain"],
            batch["mode"],
        )
    # --------------------------------
    # -----  Test loaders paths  -----
    # --------------------------------
    print(
        "All Loaders: \n",
        [
            Path(loaders[mode][domain].dataset.file_list_path).stem
            for mode in loaders
            for domain in loaders[mode]
        ],
    )

    # --------------------------------------
    # -----  Test multi_batch content  -----
    # --------------------------------------
    for i, multi_batch in enumerate(
        zip(*[loaders["train"][domain] for domain in loaders["train"]])
    ):
        print(
            "\n\n".join(
                [
                    str(
                        {
                            k: [(s, t.shape) for s, t in v.items()]
                            if k == "data"
                            else v
                            for k, v in m.items()
                            if k != "paths"
                        }
                    )
                    for m in multi_batch
                ]
            )
        )
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch}

        if i > 5:
            break
