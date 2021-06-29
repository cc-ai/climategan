import sys
from argparse import ArgumentParser
from pathlib import Path
from comet_ml import Experiment

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

import climategan

GROUND_MODEL = "/miniscratch/_groups/ccai/experiments/runs/ablation-v1/out--ground"


def uint8(array):
    return array.astype(np.uint8)


def crop_and_resize(image_path, label_path):
    """
    Resizes an image so that it keeps the aspect ratio and the smallest dimensions
    is 640, then crops this resized image in its center so that the output is 640x640
    without aspect ratio distortion

    Args:
        image_path (Path or str): Path to an image
        label_path (Path or str): Path to the image's associated label

    Returns:
        tuple((np.ndarray, np.ndarray)): (new image, new label)
    """

    img = imread(image_path)
    lab = imread(label_path)

    # if img.shape[-1] == 4:
    #     img = uint8(rgba2rgb(img) * 255)

    # TODO: remove (debug)
    if img.shape[:2] != lab.shape[:2]:
        print(
            "\nWARNING: shape mismatch: im -> {}, lab -> {}".format(
                image_path.name, label_path.name
            )
        )
        # breakpoint()

    # resize keeping aspect ratio: smallest dim is 640
    h, w = img.shape[:2]
    if h < w:
        size = (640, int(640 * w / h))
    else:
        size = (int(640 * h / w), 640)

    r_img = resize(img, size, preserve_range=True, anti_aliasing=True)
    r_img = uint8(r_img)

    r_lab = resize(lab, size, preserve_range=True, anti_aliasing=False, order=0)
    r_lab = uint8(r_lab)

    # crop in the center
    H, W = r_img.shape[:2]

    top = (H - 640) // 2
    left = (W - 640) // 2

    rc_img = r_img[top : top + 640, left : left + 640, :]
    rc_lab = (
        r_lab[top : top + 640, left : left + 640, :]
        if r_lab.ndim == 3
        else r_lab[top : top + 640, left : left + 640]
    )

    return rc_img, rc_lab


def load_ground(ground_output_path, ref_image_path):
    gop = Path(ground_output_path)
    rip = Path(ref_image_path)

    ground_paths = list((gop / "eval-metrics" / "pred").glob(f"{rip.stem}.jpg")) + list(
        (gop / "eval-metrics" / "pred").glob(f"{rip.stem}.png")
    )
    if len(ground_paths) == 0:
        raise ValueError(
            f"Could not find a ground match in {str(gop)} for image {str(rip)}"
        )
    elif len(ground_paths) > 1:
        raise ValueError(
            f"Found more than 1 ground match in {str(gop)} for image {str(rip)}:"
            + f" {list(map(str, ground_paths))}"
        )
    ground_path = ground_paths[0]
    _, ground = crop_and_resize(rip, ground_path)
    ground = (ground > 0).astype(np.float32)
    return torch.from_numpy(ground).unsqueeze(0).unsqueeze(0).cuda()


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
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Comet.ml tags",
        default=["x", "d", "s", "m", "mx", "p"],
        type=str,
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

    ims = sorted(climategan.utils.find_images(ims_path), key=lambda x: x.name)
    labs = sorted(
        climategan.utils.find_images(lab_path),
        key=lambda x: x.name.replace("_labeled.", "."),
    )

    xs = climategan.transforms.PrepareInference()(ims)
    ys = climategan.transforms.PrepareInference(is_label=True)(labs)

    return xs, ys, ims, labs


def load_inferences(inf_path, im_paths):
    try:
        assert inf_path.exists()
        assert sorted([i.stem for i in im_paths]) == sorted(
            [i.stem for i in inf_path.glob("*.pt")]
        )
        return [torch.load(str(i)) for i in tqdm(list(inf_path.glob("*.pt")))]
    except Exception as e:
        print()
        print(e)
        print("Aborting Loading")
        print()
        return None


def get_or_load_inferences(
    m_path, device, xs, is_ground, im_paths, ground_model, try_load=True
):
    inf_path = Path(m_path) / "inferences"
    if try_load:
        print("Trying to load existing inferences:")
        outputs = load_inferences(inf_path, im_paths)
        if outputs is not None:
            print("Successfully loaded existing inferences")
            return outputs

    trainer = climategan.trainer.Trainer.resume_from_path(
        m_path if not is_ground else ground_model,
        inference=True,
        new_exp=None,
        device=device,
    )

    inf_path.mkdir(exist_ok=True)
    outputs = []
    for i, x in enumerate(tqdm(xs)):
        x = x.to(trainer.device)
        if not is_ground:
            out = trainer.G.decode(x=x)
        else:
            out = {"m": load_ground(GROUND_MODEL, im_paths[i])}
        out["p"] = trainer.G.paint(out["m"] > 0.5, x)
        out["x"] = x
        inference = {k: v.cpu() for k, v in out.items()}
        outputs.append(inference)
        torch.save(inference, inf_path / f"{im_paths[i].stem}.pt")
    print()

    return outputs


def numpify(outputs):
    nps = []
    print("Numpifying...")
    for o in tqdm(outputs):
        x = (o["x"][0].permute(1, 2, 0).numpy() + 1) / 2
        m = o["m"]
        m = (m[0, 0, :, :].numpy() > 0.5).astype(np.uint8)
        p = (o["p"][0].permute(1, 2, 0).numpy() + 1) / 2
        data = {"m": m, "p": p, "x": x}
        if "s" in o:
            s = climategan.data.decode_segmap_merged_labels(o["s"], "r", False) / 255.0
            data["s"] = s[0].permute(1, 2, 0).numpy()
        if "d" in o:
            d = climategan.tutils.normalize_tensor(o["d"]).squeeze().numpy()
            data["d"] = d
        nps.append({k: img_as_ubyte(v) for k, v in data.items()})
    return nps


def concat_npy_for_model(data, tasks):
    assert "m" in data
    assert "x" in data
    assert "p" in data

    x = mask = depth = seg = painted = masked = None

    x = data["x"]
    painted = data["p"]
    mask = (gray2rgb(data["m"]) * 255).astype(np.uint8)
    painted = data["p"]
    masked = (1 - gray2rgb(data["m"])) * x

    concats = []

    if "d" in data:
        depth = img_as_ubyte(
            gray2rgb(
                resize(data["d"], data["x"].shape[:2], anti_aliasing=True, order=1)
            )
        )
    else:
        depth = np.ones_like(data["x"]) * 255

    if "s" in data:
        seg = img_as_ubyte(
            resize(data["s"], data["x"].shape[:2], anti_aliasing=False, order=0)
        )
    else:
        seg = np.ones_like(data["x"]) * 255

    for t in tasks:
        if t == "x":
            concats.append(x)
        if t == "m":
            concats.append(mask)
        elif t == "mx":
            concats.append(masked)
        elif t == "d":
            concats.append(depth)
        elif t == "s":
            concats.append(seg)
        elif t == "p":
            concats.append(painted)

    row = np.concatenate(concats, axis=1)

    return row


if __name__ == "__main__":
    args = parse_args()

    with open(args.yaml, "r") as f:
        maskers = yaml.safe_load(f)
    if "models" in maskers:
        maskers = maskers["models"]

    load = not args.disable_loading
    tags = args.tags
    tasks = args.tasks

    ground_model = None
    for m in maskers:
        if "ground" not in maskers:
            ground_model = m
            break
    if ground_model is None:
        raise ValueError("Could not find a non-ground model to get a painter")

    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

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

    exp = Experiment(project_name="climategan-inferences", display_summary_level=0)
    exp.log_parameter("names", names)
    exp.add_tags(tags)

    for i in tqdm(range(len(xs))):
        all_models_for_image = []
        for name in names:
            xpmds = concat_npy_for_model(np_outs[name][i], tasks)
            all_models_for_image.append(xpmds)
        full_im = np.concatenate(all_models_for_image, axis=0)
        pil_im = Image.fromarray(full_im)
        exp.log_image(pil_im, name=im_paths[i].stem.replace(".", "_"), step=i)
