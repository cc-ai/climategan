from pathlib import Path
import json
import datetime
import os


def make_kitti_jsons(
    base,
    out_dir,
    val_scene="/Scene05/",
    keep_frames=[
        "15-deg-left",
        "15-deg-right",
        "30-deg-left",
        "30-deg-right",
        "clone",
        "overcast",
    ],
):
    """
    https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2

    SceneX/Y/frames/rgb/Camera_Z/rgb_%05d.jpg
    SceneX/Y/frames/depth/Camera_Z/depth_%05d.png
    SceneX/Y/frames/classsegmentation/Camera_Z/classgt_%05d.png
    SceneX/Y/frames/instancesegmentation/Camera_Z/instancegt_%05d.png
    SceneX/Y/frames/backwardFlow/Camera_Z/backwardFlow_%05d.png
    SceneX/Y/frames/backwardSceneFlow/Camera_Z/backwardSceneFlow_%05d.png
    SceneX/Y/frames/forwardFlow/Camera_Z/flow_%05d.png
    SceneX/Y/frames/forwardSceneFlow/Camera_Z/sceneFlow_%05d.png
    SceneX/Y/colors.txt
    SceneX/Y/extrinsic.txt
    SceneX/Y/intrinsic.txt
    SceneX/Y/info.txt
    SceneX/Y/bbox.txt
    SceneX/Y/pose.txt

    where X ∈ {01, 02, 06, 18, 20} and represent one of 5 different locations.
    Y ∈ {15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning,
        overcast, rain, sunset} and represent the different variations.
    Z ∈ [0, 1] and represent the left (same as in virtual kitti) or right camera
        (offset by 0.532725m to the right).

    Note that our indexes always start from 0.
    """

    path = Path(base).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    depths = (path / "depth").glob("**/*.png")
    segs = (path / "segmentation").glob("**/*.png")
    rgbs = (path / "rgb").glob("**/*.jpg")
    train_data = []
    val_data = []

    for i, (x, s, d) in enumerate(zip(rgbs, segs, depths)):
        if i % 100 == 0:
            print(i, end="\r", flush=True)
        item = {"x": str(x), "s": str(s), "d": str(d)}
        if val_scene in str(x):
            val_data.append(item)
        else:
            train_data.append(item)

    with (out / "train_kitti.json").open("w") as f:
        json.dump(train_data, f)
    with (out / "val_kitti.json").open("w") as f:
        json.dump(val_data, f)
    with (out / "about_kitti_jsons.txt").open("w") as f:
        f.write("{}: {}\n".format("val_scene", val_scene))
        f.write("{}: {}\n".format("keep_frames", keep_frames))
        f.write("Author: {}\n".format(os.environ.get("USER")))
        f.write("Date: {}\n".format(str(datetime.datetime.now())))


if __name__ == "__main__":
    kitti_dir = "/miniscratch/_groups/ccai/data/vkitti2"
    out_dir = "/miniscratch/_groups/ccai/data/jsons"

    val_scene_id = "/Scene05/"

    keep_frames = [
        "15-deg-left",
        "15-deg-right",
        "30-deg-left",
        "30-deg-right",
        "clone",
        "overcast",
    ]

    make_kitti_jsons(
        kitti_dir, out_dir, val_scene=val_scene_id, keep_frames=keep_frames,
    )
