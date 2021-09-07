import torch
import torch.nn.functional as F
import random
import kornia
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

from climategan.tutils import normalize, retrieve_sky_mask

try:
    from kornia.filters import filter2d
except ImportError:
    from kornia.filters import filter2D as filter2d


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

    n_lines = int(p_h * mask.shape[-2])
    n_cols = int(p_w * mask.shape[-1])

    temp_mask = mask.clone().detach()
    for i in range(1, n_cols):
        temp_mask[:, :, :, i::] += mask[:, :, :, 0:-i]
        temp_mask[:, :, :, 0:-i] += mask[:, :, :, i::]

    new_mask = temp_mask.clone().detach()
    for i in range(1, n_lines):
        new_mask[:, :, i::, :] += temp_mask[:, :, 0:-i, :]
        new_mask[:, :, 0:-i, :] += temp_mask[:, :, i::, :]

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


def add_fire(x, seg_preds, fire_opts):
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
    wildfire_tens = normalize(x, 0, 255)

    # Warm the image
    wildfire_tens[:, 2, :, :] -= 20
    wildfire_tens[:, 1, :, :] -= 10
    wildfire_tens[:, 0, :, :] += 40
    wildfire_tens.clamp_(0, 255)
    wildfire_tens = wildfire_tens.to(torch.uint8)

    # Darken the picture and increase contrast
    wildfire_tens = adjust_contrast(wildfire_tens, contrast_factor=1.5)
    wildfire_tens = adjust_brightness(wildfire_tens, brightness_factor=0.73)

    sky_mask = retrieve_sky_mask(seg_preds).unsqueeze(1)

    if fire_opts.get("crop_bottom_sky_mask"):
        i = 2 * sky_mask.shape[-2] // 3
        sky_mask[..., i:, :] = 0

    sky_mask = F.interpolate(
        sky_mask.to(torch.float),
        (wildfire_tens.shape[-2], wildfire_tens.shape[-1]),
    )
    sky_mask = increase_sky_mask(sky_mask, 0.18, 0.18)

    kernel_size = (fire_opts.get("kernel_size", 301), fire_opts.get("kernel_size", 301))
    sigma = (fire_opts.get("kernel_sigma", 150.5), fire_opts.get("kernel_sigma", 150.5))
    border_type = "reflect"
    kernel = torch.unsqueeze(
        kornia.filters.kernels.get_gaussian_kernel2d(kernel_size, sigma), dim=0
    ).to(x.device)
    sky_mask = filter2d(sky_mask, kernel, border_type)

    filter_ = torch.ones(wildfire_tens.shape, device=x.device)
    filter_[:, 0, :, :] = 255
    filter_[:, 1, :, :] = random.randint(100, 150)
    filter_[:, 2, :, :] = 0

    wildfire_tens = paste_tensor(wildfire_tens, filter_, sky_mask, 200)

    wildfire_tens = adjust_brightness(wildfire_tens.to(torch.uint8), 0.8)
    wildfire_tens = wildfire_tens.to(torch.float)

    # dummy pixels to fool scaling and preserve range
    wildfire_tens[:, :, 0, 0] = 255.0
    wildfire_tens[:, :, -1, -1] = 0.0

    return wildfire_tens


def paste_tensor(source, filter_, mask, transparency):
    mask = transparency / 255.0 * mask
    new = mask * filter_ + (1.0 - mask) * source
    return new
