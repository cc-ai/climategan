import sys
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from argparse import ArgumentParser

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
    assert img.shape == lab.shape

    # resize keeping aspect ratio: smallest dim is 640
    h, w = img.shape[:2]
    if h < w:
        size = (640, int(640 * w / h))
    else:
        size = (int(640 * h / w), 640)

    r_img = resize(img, size)
    r_lab = resize(lab, size, order=0)  # nearest neighbor for labels

    # crop in the center
    H, W = r_img.shape[:2]

    top = (H - 640) // 2
    left = (W - 640) // 2

    rc_img = r_img[top : top + 640, left : left + 640, :]
    rc_lab = r_lab[top : top + 640, left : left + 640, :]

    return rc_img, rc_lab


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
    args = parser.parse_args()

    input_base = Path(args.input_dir).expanduser().resolve()
    output_base = Path(args.output_dir).expanduser().resolve()

    input_labels = input_base / "labels"
    input_images = input_base / "imgs"
    output_labels = output_base / "labels"
    output_images = output_base / "imgs"

    print("Input images:", str(input_images))
    print("Input labels:", str(input_labels))
    print("Output images:", str(output_images))
    print("Output labels:", str(output_labels))
    print()

    assert input_labels.exists()
    assert input_images.exists()
    if output_base.exists():
        if (
            "n"
            in input(
                "WARNING: output dir already exists."
                + " Overwrite its content? (y/n, default: y)"
            ).lower()
        ):
            sys.exit()

    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    images_paths = list(map(Path, sorted((map(str, find_images(input_images))))))
    labels_paths = list(map(Path, sorted((map(str, find_images(input_labels))))))

    for i, (image_path, label_path) in enumerate(zip(images_paths, labels_paths)):
        print(
            f"Processing {i + 1 :3} / {len(images_paths)} : {image_path.name}",
            end="\r",
            flush=True,
        )
        resized_image, resized_label = crop_and_resize(image_path, label_path)
        imsave(output_images / f"{image_path.stem}.png", resized_image)
        imsave(output_labels / f"{label_path.stem}.png", resized_label)

    print("\nDone.")
