import argparse
import os
import sys
import uuid
from pathlib import Path

import addict

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import get_all_loaders
from omnigan.utils import (
    env_to_path,
    flatten_opts,
    get_increased_path,
    load_test_opts,
)
from omnigan.tutils import domains_to_class_tensor
from run import print_header


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":

    print_header("test_domains_to_class_tensor")

    opts = opts.copy()
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    loaders = get_all_loaders(opts)

    # ---------------------------------------------
    # -----  Testing domains_to_class_tensor  -----
    # ---------------------------------------------
    batch = next(iter(loaders["train"]["r"]))
    print(domains_to_class_tensor(batch["domain"], True))
    print(domains_to_class_tensor(batch["domain"], False))
    domains = ["r", "s"]

    try:
        domains_to_class_tensor([1, "sg"])
        raise TypeError("Should raise a ValueError")
    except ValueError:
        print("ok.")

    # ---------------------------------
    # -----  Testing env_to_path  -----
    # ---------------------------------
    print_header("test_env_to_path")
    assert env_to_path("$HOME") == os.environ["HOME"]
    assert env_to_path("$HOME/") == os.environ["HOME"] + "/"
    assert env_to_path("$HOME/Documents") == str(Path(os.environ["HOME"]) / "Documents")
    print("ok.")

    # ----------------------------------------
    # -----  Testing get_increased_path  -----
    # ----------------------------------------
    print_header("test_get_increased_path")
    uid = str(uuid.uuid4())
    p = Path() / uid
    p.mkdir()
    get_increased_path(p).mkdir()
    get_increased_path(p).mkdir()
    get_increased_path(p).mkdir()
    paths = {str(d) for d in Path().glob(uid + "*")}
    target = {str(p), str(p) + " (1)", str(p) + " (2)", str(p) + " (3)"}
    assert paths == target
    print("ok.")
    for d in Path().glob(uid + "*"):
        d.rmdir()

    # ----------------------------------
    # -----  Testing flatten_opts  -----
    # ----------------------------------
    print_header("test_flatten_opts")
    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    assert flatten_opts(d) == {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }
    print("ok.")
