from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from comet_ml import Experiment
from skimage.transform import resize
from skimage.color import gray2rgb
import torch
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from ... import eval_masker
import omnigan


GROUND_MODEL = "/miniscratch/_groups/ccai/experiments/runs/ablation-v1/out--ground"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-y", "--yaml", help="Path to a list of models")
    parser.add_argument(
        "--disable_loading",
        action="store_true",
        default=False,
        help="Disable loading of existing inferences",
    )
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args = parser.parse_args()

    print("Received args:")
    print(vars(args))

    return args


def load_images_and_labels(
    path="/miniscratch/_groups/ccai/data/omnigan/masker-test-set",
):
    p = Path(path)
    ims_path = p / "imgs"
    lab_path = p / "labels"

    ims = sorted(omnigan.utils.find_images(ims_path), key=lambda x: x.name)
    labs = sorted(
        omnigan.utils.find_images(lab_path),
        key=lambda x: x.name.replace("_labeled.", "."),
    )

    xs = omnigan.transforms.PrepareInference()(ims)
    ys = omnigan.transforms.PrepareInference(is_label=True)(labs)

    return xs, ys, ims, labs


def load_inferences(inf_path, im_paths):
    try:
        assert inf_path.exists()
        assert sorted([i.stem for i in im_paths]) == sorted(
            [i.stem for i in inf_path.glob("*.pt")]
        )
        return [torch.load(str(i)) for i in inf_path.glob("*.pt")]
    except Exception as e:
        print()
        print(e)
        print("Aborting Loading")
        return None


def get_or_load_inferences(
    m_path, device, xs, is_ground, im_paths, ground_model, try_load=True
):
    inf_path = Path(m_path) / "inferences"
    if try_load:
        outputs = load_inferences(m_path, im_paths)
        if outputs is not None:
            print("Successfully loaded existing inferences")
            return outputs

    trainer = omnigan.trainer.Trainer.resume_from_path(
        m_path if not is_ground else ground_model,
        inference=True,
        new_exp=None,
        device=device,
    )

    inf_path.mkdir(exist_ok=True)
    outputs = []
    for i, x in enumerate(xs):
        x = x.to(trainer.device)
        print(i, end=" | ")
        if not is_ground:
            out = trainer.G.decode(x=x)
        else:
            out = {"m": eval_masker.load_ground(GROUND_MODEL, im_paths[i])}
        out["p"] = trainer.G.paint(out["m"] > 0.5, x)
        out["x"] = x
        inference = {k: v.cpu() for k, v in out.items()}
        outputs.append(inference)
        torch.save(inference, inf_path / f"{im_paths[i].stem}.pt")
    print()

    return outputs


def numpify(outputs):
    nps = []
    for k, o in outputs:
        print(o, end=" | ")
        x = (o["x"][0].permute(1, 2, 0).numpy() + 1) / 2
        m = o["m"]
        m = (m[0, 0, :, :].numpy() > 0).astype(np.float32)
        p = (o["p"][0].permute(1, 2, 0).numpy() + 1) / 2
        data = {"m": m, "p": p, "x": x}
        if "s" in o:
            s = omnigan.data.decode_segmap_merged_labels(o["s"], "r", False) / 255.0
            data["s"] = s
        if "d" in o:
            d = omnigan.tutils.normalize_tensor(o["d"]).numpy()
            data["d"] = d
        nps.append(data)
    print()
    return nps


def concat_npy_for_model(data):
    assert "m" in data
    assert "x" in data
    assert "p" in data

    xpm = np.concatenate([data["x"], data["p"], (1 - data["m"]) * data["x"]], axis=1)

    if "d" in data:
        depth = gray2rgb(resize(data["d"], anti_aliasing=False, order=0))
    else:
        depth = np.ones_like(data["x"], dtype=np.float32)
    xpmd = np.concatenate([xpm, depth], axis=1)

    if "s" in data:
        seg = resize(data["s"], anti_aliasing=False, order=0)
    else:
        seg = np.ones_like(data["x"], dtype=np.float32)
    xpmds = np.concatenate([xpmd, seg], axis=1)

    return xpmds


if __name__ == "__main__":
    args = parse_args()

    with open(args.yaml, "r") as f:
        maskers = yaml.safe_load(f)
    if "models" in maskers:
        maskers = maskers["models"]

    load = not args.disable_loading
    tags = args.tags

    ground_model = None
    for m in maskers:
        if "ground" not in maskers:
            ground_model = m
            break
    if ground_model is None:
        raise ValueError("Could not find a non-ground model to get a painter")

    device = torch.device("cuda:0")

    xs, ys, im_paths, lab_paths = load_images_and_labels()

    np_outs = {}
    names = []

    for m_path in maskers:

        opt_path = Path(m_path) / "opts.yaml"
        with opt_path.open("r") as f:
            opt = yaml.safe_load(f)

        name = (
            ", ".join(
                [
                    t
                    for t in sorted(opt["comet"]["tags"])
                    if "branch" not in t and "ablation" not in t and "trash" not in t
                ]
            )
            if "--ground" not in m_path
            else "ground"
        )
        names.append(name)

        is_ground = name == "ground"

        print("#" * 100)
        print("\n>>> Processing", name)
        print()

        outputs = get_or_load_inferences(
            m_path, device, xs, is_ground, im_paths, ground_model, load
        )
        nps = numpify(outputs)

        np_outs[name] = nps

    exp = Experiment(project_name="omnigan-inferences", display_summary_level=0)
    exp.log_parameter("names", names)
    exp.add_tags(tags)

    for i in range(len(xs)):
        all_models_for_image = []
        for name in names:
            xpmds = concat_npy_for_model(np_outs[name])
            all_models_for_image.append(xpmds)
        full_im = np.concatenate(all_models_for_image, axis=0)
        exp.log_image((full_im * 255).astype(np.uint8), name=im_paths[i], step=i)
