import json
import os
from pathlib import Path

if __name__ == "__main__":
    # labelbox json export path
    path = "/Users/victor/Downloads/export-2021-02-27T17_15_30.291Z.json"
    # where to write the downloaded images
    out = Path("/Users/victor/Downloads/labelbox_test_flood-v2")
    # create out dir
    out.mkdir(exist_ok=True, parents=True)

    # load export data
    with open(path, "r") as f:
        data = json.load(f)

    for i, d in enumerate(data):
        # find all polygons
        objects = d["Label"]["objects"]
        # retrieve original image name
        name = d["External ID"]
        stem = Path(name).stem
        # output dir for current image
        m_out = out / stem[:30]
        m_out.mkdir(exist_ok=True, parents=True)

        # save 1 png per polygon
        for o, obj in enumerate(objects):
            print(f"{i}/{len(data)} : {o}/{len(objects)}")

            # create verbose label -> "cannotflood", "mustflood"
            label = obj["value"].replace("_", "")
            # unique polygon mask filename
            m_path = m_out / f"{stem}_{label}_{o}.png"
            # download address for curl
            uri = obj["instanceURI"]
            # command to download the image
            command = f'curl {uri} > "{str(m_path)}"'
            # execute command
            os.system(command)
            print("#" * 20)
