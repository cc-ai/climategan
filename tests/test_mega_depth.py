import sys
from pathlib import Path
from skimage import io
from addict import Dict

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from run import opts, print_header

from omnigan.mega_depth import get_mega_model
from omnigan.utils import decode_mega_depth
from omnigan.data import get_all_loaders
from omnigan.trainer import Trainer


if __name__ == "__main__":

    opts = opts.copy()
    crop_to = 32
    write_images = True
    test_batch = False
    test_translation = True

    not_committed_path = Path(__file__).parent / "not_committed"

    if not not_committed_path.exists():
        not_committed_path.mkdir()

    if crop_to > 0:
        opts.data.transforms += [
            Dict({"name": "crop", "ignore": False, "height": crop_to, "width": crop_to})
        ]

    mega = get_mega_model()
    loaders = get_all_loaders(opts)
    loader = loaders["train"]["rn"]
    batch = next(iter(loader))

    if test_batch:
        print_header("infer MD on batch")
        im_t = batch["data"]["x"]
        print("inferring...")
        im_d = mega(im_t)
        print("Done. Saving...")
        for i, im in enumerate(im_d):
            im_n = decode_mega_depth(im, numpy=True)
            stem = Path(batch["paths"]["s"][i]).stem
            if write_images:
                io.imsave(
                    str(not_committed_path / (stem + "_depth.png")), im_n,
                )
        print("Done.")

    if test_translation:
        print_header("translate then infer MD")
        trainer = Trainer(opts, verbose=1)
        trainer.setup()
        print("Translating...")
        y = trainer.G.translate_batch(batch)
        print("Done. Inferring depth...")
        y_d = mega(y)
        print("Done.")
        if write_images:
            for i, im_d in enumerate(y_d):
                print(i, "/", len(y_d))
                im_n = decode_mega_depth(im_d, numpy=True)
                stem = Path(batch["paths"]["s"][i]).stem
                io.imsave(
                    str(not_committed_path / (stem + "_translated_depth.png")), im_n,
                )
        else:
            im_d = decode_mega_depth(y_d)
