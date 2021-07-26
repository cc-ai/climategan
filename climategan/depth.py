import torch
import torch.nn as nn
import torch.nn.functional as F

from climategan.blocks import BaseDecoder, Conv2dBlock, InterpolateNearest2d
from climategan.utils import find_target_size


def create_depth_decoder(opts, no_init=False, verbose=0):
    if opts.gen.d.architecture == "base":
        decoder = BaseDepthDecoder(opts)
        if "s" in opts.task:
            assert opts.gen.s.use_dada is False
        if "m" in opts.tasks:
            assert opts.gen.m.use_dada is False
    else:
        decoder = DADADepthDecoder(opts)

    if verbose > 0:
        print(f"  - Add {decoder.__class__.__name__}")

    return decoder


class DADADepthDecoder(nn.Module):
    """
    Depth decoder based on depth auxiliary task in DADA paper
    """

    def __init__(self, opts):
        super().__init__()
        if (
            opts.gen.encoder.architecture == "deeplabv3"
            and opts.gen.deeplabv3.backbone == "mobilenet"
        ):
            res_dim = 320
        else:
            res_dim = 2048

        mid_dim = 512

        self.do_feat_fusion = False
        if opts.gen.m.use_dada or ("s" in opts.tasks and opts.gen.s.use_dada):
            self.do_feat_fusion = True
            self.dec4 = Conv2dBlock(
                128,
                res_dim,
                1,
                stride=1,
                padding=0,
                bias=True,
                activation="lrelu",
                norm="none",
            )

        self.relu = nn.ReLU(inplace=True)
        self.enc4_1 = Conv2dBlock(
            res_dim,
            mid_dim,
            1,
            stride=1,
            padding=0,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.enc4_2 = Conv2dBlock(
            mid_dim,
            mid_dim,
            3,
            stride=1,
            padding=1,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.enc4_3 = Conv2dBlock(
            mid_dim,
            128,
            1,
            stride=1,
            padding=0,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.upsample = None
        if opts.gen.d.upsample_featuremaps:
            self.upsample = nn.Sequential(
                *[
                    InterpolateNearest2d(),
                    Conv2dBlock(
                        128,
                        32,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                        activation="lrelu",
                        pad_type="reflect",
                        norm="batch",
                    ),
                    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                ]
            )
        self._target_size = find_target_size(opts, "d")
        print(
            "      - {}:  setting target size to {}".format(
                self.__class__.__name__, self._target_size
            )
        )

    def set_target_size(self, size):
        """
        Set final interpolation's target size

        Args:
            size (int, list, tuple): target size (h, w). If int, target will be (i, i)
        """
        if isinstance(size, (list, tuple)):
            self._target_size = size[:2]
        else:
            self._target_size = (size, size)

    def forward(self, z):
        if isinstance(z, (list, tuple)):
            z = z[0]
        z4_enc = self.enc4_1(z)
        z4_enc = self.enc4_2(z4_enc)
        z4_enc = self.enc4_3(z4_enc)

        z_depth = None
        if self.do_feat_fusion:
            z_depth = self.dec4(z4_enc)

        if self.upsample is not None:
            z4_enc = self.upsample(z4_enc)

        depth = torch.mean(z4_enc, dim=1, keepdim=True)  # DADA paper decoder
        if depth.shape[-1] != self._target_size:
            depth = F.interpolate(
                depth,
                size=(384, 384),  # size used in MiDaS inference
                mode="bicubic",  # what MiDaS uses
                align_corners=False,
            )

            depth = F.interpolate(
                depth, (self._target_size, self._target_size), mode="nearest"
            )  # what we used in the transforms to resize input

        return depth, z_depth

    def __str__(self):
        return "DADA Depth Decoder"


class BaseDepthDecoder(BaseDecoder):
    def __init__(self, opts):
        low_level_feats_dim = -1
        use_v3 = opts.gen.encoder.architecture == "deeplabv3"
        use_mobile_net = opts.gen.deeplabv3.backbone == "mobilenet"
        use_low = opts.gen.d.use_low_level_feats

        if use_v3 and use_mobile_net:
            input_dim = 320
            if use_low:
                low_level_feats_dim = 24
        elif use_v3:
            input_dim = 2048
            if use_low:
                low_level_feats_dim = 256
        else:
            input_dim = 2048

        n_upsample = 1 if opts.gen.d.upsample_featuremaps else 0
        output_dim = (
            1
            if not opts.gen.d.classify.enable
            else opts.gen.d.classify.linspace.buckets
        )

        self._target_size = find_target_size(opts, "d")
        print(
            "      - {}:  setting target size to {}".format(
                self.__class__.__name__, self._target_size
            )
        )

        super().__init__(
            n_upsample=n_upsample,
            n_res=opts.gen.d.n_res,
            input_dim=input_dim,
            proj_dim=opts.gen.d.proj_dim,
            output_dim=output_dim,
            norm=opts.gen.d.norm,
            activ=opts.gen.d.activ,
            pad_type=opts.gen.d.pad_type,
            output_activ="none",
            low_level_feats_dim=low_level_feats_dim,
        )

    def set_target_size(self, size):
        """
        Set final interpolation's target size

        Args:
            size (int, list, tuple): target size (h, w). If int, target will be (i, i)
        """
        if isinstance(size, (list, tuple)):
            self._target_size = size[:2]
        else:
            self._target_size = (size, size)

    def forward(self, z, cond=None):
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate depth to the target depth map's size"
            raise ValueError(error)

        d = super().forward(z)

        preds = F.interpolate(
            d, size=self._target_size, mode="bilinear", align_corners=True
        )

        return preds, None
