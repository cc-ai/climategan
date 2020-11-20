import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


def normalize(arr, mini=0, maxi=1):
    return mini + (maxi - mini) * (arr - arr.min()) / (arr.max() - arr.min())


def retrieve_sky_mask(seg_preds):
    seg_ind = torch.argmax(seg_preds.squeeze(), dim=0)
    sky_mask = seg_ind == 9
    return sky_mask


def increase_sky_mask(sky_mask, increase=True, p_w=0, p_h=0):
    if p_h <= 0 and p_w <= 0:
        return sky_mask

    n_lines = int(p_h * sky_mask.shape[0])
    n_cols = int(p_w * sky_mask.shape[1])

    for i in range(1, n_cols):
        sky_mask[:, i::] += sky_mask[:, 0:-i]
        sky_mask[:, 0:-i] += sky_mask[:, i::]
    for i in range(1, n_lines):
        sky_mask[i::, :] += sky_mask[0:-i, :]
        sky_mask[0:-i, :] += sky_mask[i::, :]

    sky_mask[sky_mask >= 1] = 1

    return sky_mask


def add_fire(x, seg_preds, filter_color, blur_radius):
    x_arr = (
        normalize(x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()) * 255
    ).astype(np.uint8)
    im = Image.fromarray(x_arr).convert("RGB")

    # Darken the picture and increase contrast
    contraster = ImageEnhance.Contrast(im)
    im = contraster.enhance(2.0)
    darkener = ImageEnhance.Brightness(im)
    im = darkener.enhance(0.25)

    # Make the image more red
    im_array = np.array(im)
    im_array[:, :, 2] = np.minimum(im_array[:, :, 2], im_array[:, :, 2] - 20)
    im_array[:, :, 1] = np.minimum(im_array[:, :, 1], im_array[:, :, 1] - 10)
    im_array[:, :, 0] = np.maximum(im_array[:, :, 0], im_array[:, :, 0] + 40)
    im = Image.fromarray(im_array).convert("RGB")

    # Find sky proportion in picture
    sky_mask = retrieve_sky_mask(seg_preds)
    sky_mask = F.interpolate(
        sky_mask.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor),
        (im.size[1], im.size[0]),
    )
    num_sky_pixels = torch.sum(sky_mask)
    sky_proportion = num_sky_pixels / (sky_mask.shape[0] * sky_mask.shape[1])
    has_sky = sky_proportion > 0

    # Adding red-ish color mostly in the sky
    if has_sky:
        filter_ = Image.new("RGB", im.size, filter_color)
        sky_mask = increase_sky_mask(sky_mask, True, 0.01, 0.01)
        im_mask = Image.fromarray((sky_mask.cpu().numpy() * 255.0).squeeze()).convert(
            "L"
        )
        filter_mask = im_mask.filter(ImageFilter.GaussianBlur(blur_radius))
        im.paste(filter_, (0, 0), filter_mask)

    darkener = ImageEnhance.Brightness(im)
    im = darkener.enhance(0.9)

    wildfire_tens = transforms.ToTensor()(im).unsqueeze(0)

    return wildfire_tens
