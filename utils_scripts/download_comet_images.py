import comet_ml
from pathlib import Path
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_id", type=str, required=True)
    parser.add_argument("-d", "--download_dir", type=str, default=None)
    parser.add_argument(
        "-s", "--step", default="last", type=str, help="last, all or int"
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        default="./",
        type=str,
        help="download into base_dir/exp_id[:8]/",
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir or Path(args.base_dir) / args.exp_id[:8])
    download_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(download_dir)

    print(
        "Downloading exp {}'s image at step `{}` into {}".format(
            args.exp_id, args.step, download_dir
        )
    )

    step = None
    try:
        step = int(args.step)
    except ValueError:
        step = args.step
        assert step in {"last", "all"}

    api = comet_ml.api.API()
    exp = api.get_experiment_by_id(args.exp_id)
    ims = exp.get_asset_list(asset_type="image")

    if step == "last":
        last_step = max(i["step"] for i in ims)
        ims = [i for i in ims if i["step"] == last_step]
    elif isinstance(step, int):
        ims = [i for i in ims if i["step"] == step]

    with open("./url.txt", "w") as f:
        f.write(exp.url)

    for i, im in enumerate(ims):
        print(
            "\nDownloading {}/{}: {} in {}".format(
                i + 1, len(ims), im["fileName"], download_dir
            )
        )
        os.system(im["curlDownload"] + ".png")
