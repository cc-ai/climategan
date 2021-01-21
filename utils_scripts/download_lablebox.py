import json
import os
from pathlib import Path

if __name__ == "__main__":
    path = "/Users/victor/Downloads/export-2021-01-20T20_45_24.103Z.json"
    out = Path("/Users/victor/Downloads/labelbox_test_flood")
    out.mkdir(exist_ok=True, parents=True)

    with open(path, "r") as f:
        data = json.load(f)

    for i, d in enumerate(data):
        objects = d["Label"]["objects"]
        name = d["External ID"]
        stem = Path(name).stem
        m_out = out / stem[:30]
        m_out.mkdir(exist_ok=True, parents=True)
        for o, obj in enumerate(objects):
            print(f"{i}/{len(data)} : {o}/{len(objects)}")
            label = obj["value"].replace("_", "")
            m_path = m_out / f"{stem}_{label}_{o}.png"
            uri = obj["instanceURI"]
            command = f'curl {uri} > "{str(m_path)}"'
            print(command)
            os.system(command)
            print("#" * 20)
