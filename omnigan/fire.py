import torch
import torch.nn.functional as F
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import ToTensor
from PIL import Image, ImageFilter
from omnigan.tutils import normalize, retrieve_sky_mask


def increase_sky_mask(mask, p_w=0, p_h=0):
    """
    Increases sky mask in width and height by a given pourcentage
    (Purpose: when applying Gaussian blur, there are no artifacts of blue sky behind)
    Args:
        sky_mask (torch.Tensor): Sky mask of shape (H,W)
        p_w (float): Percentage of mask width by which to increase
            the width of the sky region
        p_h (float): Percentage of mask height by which to increase
            the height of the sky region
    Returns:
        torch.Tensor: Sky mask increased given p_w and p_h
    """

    if p_h <= 0 and p_w <= 0:
        return mask

    n_lines = int(p_h * mask.shape[0])
    n_cols = int(p_w * mask.shape[1])

    temp_mask = mask.clone().detach()
    for i in range(1, n_cols):
        temp_mask[:, i::] += mask[:, 0:-i]
        temp_mask[:, 0:-i] += mask[:, i::]

    new_mask = temp_mask.clone().detach()
    for i in range(1, n_lines):
        new_mask[i::, :] += temp_mask[0:-i, :]
        new_mask[0:-i, :] += temp_mask[i::, :]

    new_mask[new_mask >= 1] = 1

    return new_mask


def paste_filter(x, filter_, mask):
    """
    Pastes a filter over an image given a mask
    Where the mask is 1, the filter is copied as is.
    Where the mask is 0, the current value is preserved.
    Intermediate values will mix the two images together.
    Args:
        x (torch.Tensor): Input tensor, range must be [0, 255]
        filer_ (torch.Tensor): Filter, range must be [0, 255]
        mask (torch.Tensor): Mask, range must be [0, 1]
    Returns:
        torch.Tensor: New tensor with filter pasted on it
    """
    assert len(x.shape) == len(filter_.shape) == len(mask.shape)
    x = filter_ * mask + x * (1 - mask)
    return x


def add_fire(x, seg_preds, filter_color, blur_radius):
    """
    Transforms input tensor given wildfires event
    Args:
        x (torch.Tensor): Input tensor
        seg_preds (torch.Tensor): Semantic segmentation predictions for input tensor
        filter_color (tuple): (r,g,b) tuple for the color of the sky
        blur_radius (float): radius of the Gaussian blur that smooths
            the transition between sky and foreground
    Returns:
        torch.Tensor: Wildfire version of input tensor
    """
    wildfire_tens = normalize(x, 0, 255).squeeze(0)

    # Warm the image
    wildfire_tens[2, :, :] -= 20
    wildfire_tens[1, :, :] -= 10
    wildfire_tens[0, :, :] += 40
    wildfire_tens[wildfire_tens > 255] = 255
    wildfire_tens[wildfire_tens < 0] = 0
    wildfire_tens = wildfire_tens.type(torch.uint8)

    # Darken the picture and increase contrast
    wildfire_tens = adjust_contrast(wildfire_tens, contrast_factor=1.5)
    wildfire_tens = adjust_brightness(wildfire_tens, brightness_factor=0.7)

    # Find sky proportion in picture
    sky_mask = retrieve_sky_mask(seg_preds)
    sky_mask = F.interpolate(
        sky_mask.unsqueeze(0).unsqueeze(0).type(torch.float),
        (wildfire_tens.shape[-2], wildfire_tens.shape[-1]),
    )
    sky_mask = sky_mask.squeeze(0).squeeze(0)
    num_sky_pixels = torch.sum(sky_mask)
    sky_proportion = num_sky_pixels / (sky_mask.shape[0] * sky_mask.shape[1])
    has_sky = sky_proportion > 0.01

    # Adding red-ish color mostly in the sky
    if has_sky:
        im_array = wildfire_tens.permute(1, 2, 0).cpu().detach().numpy()
        im = Image.fromarray(im_array).convert("RGB")

        filter_ = Image.new("RGB", im.size, filter_color)

        sky_mask = increase_sky_mask(sky_mask, 0.2, 0.2)
        im_mask = Image.fromarray((sky_mask.cpu().numpy() * 255.0).squeeze()).convert(
            "L"
        )
        filter_mask = im_mask.filter(ImageFilter.GaussianBlur(blur_radius))

        im.paste(filter_, (0, 0), filter_mask)

        wildfire_tens = (255.0 * ToTensor()(im).to(x.device)).type(torch.uint8)

    wildfire_tens = adjust_brightness(wildfire_tens, brightness_factor=0.8)
    wildfire_tens = wildfire_tens.unsqueeze(0).type(torch.float)

    return wildfire_tens
