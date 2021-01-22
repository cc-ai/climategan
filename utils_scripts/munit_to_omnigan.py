from pathlib import Path
import json

root = "/network/tmp1/ccai/data/munit_dataset"
lists = """
data_folder_train_a: ./
data_list_train_a: /network/tmp1/ccai/data/munit_dataset/trainA.txt
data_folder_test_a: ./
data_list_test_a: /network/tmp1/ccai/data/munit_dataset/testA.txt
data_folder_train_b: ./
data_list_train_b: /network/tmp1/ccai/data/munit_dataset/trainB.txt
data_folder_test_b: ./
data_list_test_b: /network/tmp1/ccai/data/munit_dataset/testB.txt
data_list_train_a_seg: /network/tmp1/ccai/data/munit_dataset/trainA_seg.txt
data_list_train_b_seg: /network/tmp1/ccai/data/munit_dataset/trainB_seg.txt
data_list_train_a_synth: /network/tmp1/ccai/data/munit_dataset/simdata/Unity1000R_broken_water/txt_files/normal.txt
data_list_train_b_synth: /network/tmp1/ccai/data/munit_dataset/simdata/Unity1000R_broken_water/txt_files/flood.txt
data_list_train_b_seg_synth: /network/tmp1/ccai/data/munit_dataset/simdata/Unity1000R_broken_water/txt_files/mask.txt #binary mask
seg_list_a: /network/tmp1/ccai/data/munit_dataset/simdata/Unity1000R_broken_water/txt_files/seg.txt
seg_list_b: /network/tmp1/ccai/data/munit_dataset/simdata/Unity1000R_broken_water/txt_files/seg_flood.txt
"""


def get_lines(path):
    if path is not None:
        with path.open("r") as f:
            return list(map(lambda x: x.strip(), f.readlines()))


if __name__ == "__main__":
    name = "from_FeatureDA+seg"
    path_to_omnigan_data = Path(__file__).parent / "shared"
    munit_data_lists = {"TODO": None}
    for l in lists.split("\n"):
        if ":" not in l:
            continue
        k, v = l.split(":")
        if "data_folder" in k:
            continue
        munit_data_lists[k.strip()] = Path(v.split("#")[0].strip())
    # munit_data_lists = {k: Path(root) / v for k, v in munit_data_lists.items()}

    omnigan_data_lists = {
        "train_rf": {
            "x": munit_data_lists["data_list_train_b"],
            "s": munit_data_lists["TODO"],
        },
        "train_rn": {
            "x": munit_data_lists["data_list_train_a"],
            "s": munit_data_lists["TODO"],
        },
        "train_sf": {
            "x": munit_data_lists["data_list_train_b_synth"],
            "s": munit_data_lists["seg_list_b"],
        },
        "train_sn": {
            "x": munit_data_lists["data_list_train_a_synth"],
            "s": munit_data_lists["seg_list_a"],
        },
        "val_rf": {
            "x": munit_data_lists["data_list_test_b"],
            "s": munit_data_lists["TODO"],
        },
        "val_rn": {
            "x": munit_data_lists["data_list_test_a"],
            "s": munit_data_lists["TODO"],
        },
        # "val_sf": {"x": munit_data_lists["data_list_test_b_synth"], "s": munit_data_lists["data_list_train_b"]},
        # "val_sn": {"x": munit_data_lists["data_list_test_a_synth"], "s": munit_data_lists["data_list_train_b"]},
    }

    omnigan_data = {k: get_lines(v) for k, v in omnigan_data_lists.items()}

    for k, v in omnigan_data.items():
        with (path_to_omnigan_data / (f"{name}_{k}.json")).open("w") as f:
            json.dump(v, f)
