import sys
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgba2rgb
from argparse import ArgumentParser
import numpy as np

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


def is_image_file(filename):
    """Check that a file's name points to a known image format
    """
    if isinstance(filename, Path):
        return filename.suffix in IMG_EXTENSIONS

    return Path(filename).suffix in IMG_EXTENSIONS


def find_images(path, recursive=False):
    """
    Get a list of all images contained in a directory:

    - path.glob("*") if not recursive
    - path.glob("**/*") if recursive
    """
    p = Path(path)
    assert p.exists()
    assert p.is_dir()
    pattern = "*"
    if recursive:
        pattern += "*/*"

    return [i for i in p.glob(pattern) if i.is_file() and is_image_file(i)]


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
    dolab = label_path is not None

    img = imread(image_path)
    if dolab:
        lab = imread(label_path)

    if img.shape[-1] == 4:
        img = uint8(rgba2rgb(img) * 255)

    if dolab and img.shape != lab.shape:
        print("\nWARNING: shape mismatch. Entering breakpoint to investigate:")
        breakpoint()

    # resize keeping aspect ratio: smallest dim is 640
    h, w = img.shape[:2]
    if h < w:
        size = (640, int(640 * w / h))
    else:
        size = (int(640 * h / w), 640)

    r_img = resize(img, size, preserve_range=True, anti_aliasing=True)
    r_img = uint8(r_img)

    if dolab:
        # nearest neighbor for labels
        r_lab = resize(lab, size, preserve_range=True, anti_aliasing=False, order=0)
        r_lab = uint8(r_lab)

    # crop in the center
    H, W = r_img.shape[:2]

    top = (H - 640) // 2
    left = (W - 640) // 2

    rc_img = r_img[top : top + 640, left : left + 640, :]
    if dolab:
        rc_lab = r_lab[top : top + 640, left : left + 640, :]
    else:
        rc_lab = None

    return rc_img, rc_lab


def label(img, label, alpha=0.4):
    return uint8(alpha * label + (1 - alpha) * img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", type=str, help="Directory to recursively read images from"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Where to writ the result of the script,"
        + " keeping the input dir's structure",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Only process images, don't look for labels",
    )
    parser.add_argument(
        "--store_labeled",
        action="store_true",
        help="Store a superposition of the label and the image in out/labeled/",
    )
    args = parser.parse_args()

    dolab = not args.no_labels
    dolabeled = args.store_labeled

    input_base = Path(args.input_dir).expanduser().resolve()
    output_base = Path(args.output_dir).expanduser().resolve()

    input_images = input_base / "imgs"
    output_images = output_base / "imgs"

    if dolab:
        input_labels = input_base / "labels"
        output_labels = output_base / "labels"
        if dolabeled:
            output_labeled = output_base / "labeled"

    print("Input images:", str(input_images))
    print("Output images:", str(output_images))
    if dolab:
        print("Input labels:", str(input_labels))
        print("Output labels:", str(output_labels))
        if dolabeled:
            print("Output labeled:", str(output_labeled))
    else:
        print("NO LABEL PROCESSING (args.no_labels is specified)")
    print()

    assert input_images.exists()
    if dolab:
        assert input_labels.exists()

    if output_base.exists():
        if (
            "n"
            in input(
                "WARNING: output dir already exists."
                + " Overwrite its content? (y/n, default: y)"
            ).lower()
        ):
            sys.exit()

    output_images.mkdir(parents=True, exist_ok=True)
    if dolab:
        output_labels.mkdir(parents=True, exist_ok=True)
        if dolabeled:
            output_labeled.mkdir(parents=True, exist_ok=True)

    images_paths = list(
        map(Path, sorted((map(str, find_images(input_images, recursive=True)))))
    )
    if dolab:
        labels_paths = list(
            map(Path, sorted((map(str, find_images(input_labels, recursive=True)))))
        )
    else:
        labels_paths = [None] * len(images_paths)

    for i, (image_path, label_path) in enumerate(zip(images_paths, labels_paths)):
        print(
            f"Processing {i + 1 :3} / {len(images_paths)} : {image_path.name}",
            end="\r",
            flush=True,
        )
        processed_image, processed_label = crop_and_resize(image_path, label_path)
        imsave(output_images / f"{image_path.stem}.png", processed_image)
        if dolab:
            imsave(output_labels / f"{label_path.stem}.png", processed_label)
            if dolabeled:
                labeled = label(processed_image, processed_label)
                imsave(output_labeled / f"{image_path.stem}.png", labeled)

    print("\nDone.")
