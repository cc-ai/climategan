import argparse
import glob
import os
import random
import json

random.seed(1332)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    default="./example_data/flood_real_dataset/",
    type=str,
    help="Path to the data folder, a folder for each category",
)
parser.add_argument(
    "--train_size",
    default=0,
    type=int,
    help="Size of the train set in percentage, default no split",
)
opts = parser.parse_args()
data_path = opts.path
categories = {}
sample_list = {}

# Mapping data category to json index
mapping = {"Segmentation": "s", "Depth": "d", "Normal": "x"}

# List the files for every category
if os.path.exists(data_path):
    for key in mapping.keys():
        categories[key] = [f for f in glob.glob(data_path + key + "/*")]
else:
    raise ValueError("Not a correct path")

# Build dict with samples as keys and dicts of datapath as values
for category in categories.keys():
    for sample in categories[category]:
        sample_with_extension = os.path.basename(sample)
        sample_name = os.path.splitext(os.path.basename(sample_with_extension))[0]
        if sample_name not in sample_list.keys():
            sample_list[sample_name] = {}
        sample_list[sample_name][mapping[category]] = (
            data_path + category + "/" + sample_with_extension
        )


# If we want to split the dataset in train and validation set
if opts.train_size > 0:
    train_set = {}
    # Select indexes for the sampling
    train_keys = random.sample(
        sample_list.keys(), int(len(sample_list) * opts.train_size / 100)
    )
    print("Using {} images as train set".format(len(train_keys)))
    val_keys = list(sample_list.keys() - train_keys)
    print("Using {} images as val set".format(len(val_keys)))

    for key in train_keys:
        train_set[key] = sample_list[key]

    for key in train_keys:
        sample_list.pop(key)

    val_set = sample_list
    print("Creating JSON file ...")

    with open("./example_data/train.json", "w") as f:
        json.dump(train_set, f, indent=4, sort_keys=True)

    with open("./example_data/val.json", "w") as f:
        json.dump(val_set, f, indent=4, sort_keys=True)

else:
    print("The dataset contains {0} samples".format(len(sample_list)))
    print("Creating JSON file ..")

    with open("./example_data/data.json", "w") as f:
        json.dump(sample_list, f, indent=4, sort_keys=True)

