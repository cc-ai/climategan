import json
import shutil
from pathlib import Path
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    jpath = Path(args.json).expanduser().resolve()
    assert jpath.exists()

    with jpath.open("r") as f:
        data = json.load(f)

    out_path = Path(os.getenv("SLURM_TMPDIR"))
    new_json = out_path / jpath.name

    new_data = []
    file_stems = set()

    for d in data:
        new_item = {}
        for task, path in d.items():
            name = Path(path).name
            new_path = out_path / name

            i = 1
            while new_path.stem not in file_stems:
                new_path = new_path.parent / (new_path.stem + f"_{i}" + new_path.suffix)
                i += 1
            file_stems.add(new_path.stem)

            shutil.copyfile(str(path), str(new_path))

            new_item[task] = str(new_path)
        new_data.append(new_item)

    with new_json.open("w") as f:
        json.dump(new_data, f)

    print(f"Copied {len(data)} samples to {str(out_path)}\nWrote {str(new_json)}")
