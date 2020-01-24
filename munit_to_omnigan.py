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
data_list_train_a_synth: /network/tmp1/ccai/data/Unity1000R/txt_files/normal.txt
data_list_train_b_synth: /network/tmp1/ccai/data/Unity1000R/txt_files/flood.txt
data_list_train_b_seg_synth: /network/tmp1/ccai/data/Unity1000R/txt_files/mask.txt #binary mask
seg_list_a: /network/tmp1/ccai/data/Unity1000R/txt_files/seg.txt
seg_list_b: /network/tmp1/ccai/data/Unity1000R/txt_files/seg_flood.txt
"""


def get_lines(path):
    with path.open("r") as f:
        return list(map(lambda x: x.strip(), f.readlines()))


if __name__ == "__main__":
    name = "from_FeatureDA+seg"
    path_to_omnigan_data = Path(__file__) / "shared"
    munit_data_lists = {}
    for l in lists.split("\n"):
        if ":" not in l:
            continue
        k, v = l.split(":")
        if "data_folder" in k:
            continue
        munit_data_lists[k.strip()] = Path(v.split("#")[0].strip())
    # munit_data_lists = {k: Path(root) / v for k, v in munit_data_lists.items()}

    omnigan_data_lists = {
        "train_rf": munit_data_lists["data_list_train_b"],
        "train_rn": munit_data_lists["data_list_train_a"],
        "train_sf": munit_data_lists["data_list_train_b_synth"],
        "train_sn": munit_data_lists["data_list_train_a_synth"],
        "val_rf": munit_data_lists["data_list_test_b"],
        "val_rn": munit_data_lists["data_list_test_a"],
        # "val_sf": munit_data_lists["data_list_test_b_synth"], # inexistent
        # "val_sn": munit_data_lists["data_list_test_a_synth"], # inexistent
    }

    omnigan_data = {k: get_lines(v) for k, v in omnigan_data_lists.items()}

    for k, v in omnigan_data.items():
        with (path_to_omnigan_data / (f"{name}_{k}.json")).open("w") as f:
            json.dump(v, f)
