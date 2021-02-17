import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from comet_ml import Experiment
from omnigan.losses import MinentLoss
from omnigan.trainer import Trainer
from omnigan.utils import flatten_opts, load_opts
from torch.nn.functional import sigmoid
from torchvision import transforms as trsfs


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./shared/trainer/defaults.yaml",
        type=str,
        help="What configuration file to use to overwrite default",
    )
    parser.add_argument(
        "--default_config",
        default="./shared/trainer/defaults.yaml",
        type=str,
        help="What default file to use",
    )
    parser.add_argument(
        "--entropy_split",
        default=0.67,
        type=float,
        help="hyperparameter lambda to split the target domain",
    )
    parser.add_argument(
        "--include_sim",
        default=False,
        type=bool,
        help="Whether the output of the easy_split.json includes the original"
        + " train_sim.json in the first stage or not",
    )
    parser.add_argument(
        "--sim_path",
        default="/network/tmp1/ccai/data/omnigan/base/train_s_full.json",
        type=str,
        help="Path to training data on sim domain in stage one",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to experiment folder containing checkpoints/latest_ckpt.pth",
        required=True,
    )
    parser.add_argument(
        "--no_comet", action="store_true", help="DON'T use comet.ml to log experiment"
    )
    return parser.parse_args()


def tupleList2DictList(tuples, keys=["x", "m"]):
    DictList = []
    for Tuple in tuples:
        tmpDict = {}
        for i in range(len(keys)):
            tmpDict[keys[i]] = Tuple[i]
        DictList.append(tmpDict)
    return DictList


def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, "r") as f:
            result.extend(json.load(f))

    with open("easy_split_with_original_sim.json", "w") as output_file:
        json.dump(result, output_file)


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(Path(args.config), default="./shared/trainer/defaults.yaml")
    opts.train.resume = True
    opts.data.loaders.batch_size = 1

    # ----------------------------------
    # -----  Set Comet Experiment  -----
    # ----------------------------------
    exp = None
    if not args.no_comet:
        exp = Experiment(project_name="omnigan", auto_metric_logging=False)
        exp.log_parameters(flatten_opts(opts))

    # ------------------------
    # ----- Define model -----
    # ------------------------
    trainer = Trainer(opts)
    trainer.setup()
    trainer.resume()
    trainer_loader = trainer.train_loaders

    # -------------------------------
    # -----  Transforms images  -----
    # -------------------------------

    transforms = [trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entropy_loss_v2 = MinentLoss(
        version=2, lambda_var=opts.train.lambdas.advent.ent_var
    )
    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------
    entropy_list = []
    i = 0
    for multi_batch_tuple in trainer.train_loaders:
        i += 1
        if i % 100 == 0:
            print("Finished calculating " + str(i) + " th image")
        for batch in multi_batch_tuple:
            batch_domain = batch["domain"][0]
            with torch.no_grad():
                if batch_domain == "r":
                    batch = trainer.batch_to_device(batch)
                    x = batch["data"]["x"]
                    Dict = batch["paths"]  # a dict includes paths of 'x' and 'm'
                    trainer.z = trainer.G.encode(x)
                    prediction = sigmoid(trainer.G.decoders["m"](trainer.z))
                    pred_complementary = 1 - prediction
                    prob = torch.cat([prediction, pred_complementary], dim=1)
                    mask_entropy = entropy_loss_v2(prob.to(trainer.device))
                    info = []
                    for key in Dict.keys():
                        info.append(Dict[key][0])
                    info.append(mask_entropy)
                    entropy_list.append(info)
    entropy_list_sorted = entropy_list.copy()
    entropy_list_sorted = sorted(entropy_list_sorted, key=lambda img: img[2])
    entropy_rank = [(item[0], item[1]) for item in entropy_list_sorted]
    easy_split = entropy_rank[: int(len(entropy_rank) * args.entropy_split)]
    hard_split = entropy_rank[int(len(entropy_rank) * args.entropy_split) :]
    easy_splitDict = tupleList2DictList(easy_split)
    hard_splitDict = tupleList2DictList(hard_split)
    with open("easy_split.json", "w", encoding="utf-8") as outfile:
        json.dump(easy_splitDict, outfile, ensure_ascii=False)
    with open("hard_split.json", "w", encoding="utf-8") as outfile:
        json.dump(hard_splitDict, outfile, ensure_ascii=False)
    if args.include_sim and args.sim_path is not None:
        merge_JsonFiles([args.sim_path, easy_split.json])
