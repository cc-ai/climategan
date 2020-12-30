from pathlib import Path

import torch
import torch.nn as nn
from .resnet101 import ResNet101
from .resnetmulti import ResNetMulti
from .deeplabv3 import DeepLabV3Decoder  # noqa: F401
from .deeplabv2 import DeepLabV2Decoder  # noqa: F401
from .mobilenetv2 import MobileNetV2


def build_v3_backbone(opts, no_init, verbose=0):
    backbone = opts.gen.deeplabv3.backbone
    output_stride = opts.gen.deeplabv3.output_stride
    if backbone == "resnet":
        resnet = ResNet101(
            output_stride=output_stride,
            BatchNorm=nn.BatchNorm2d,
            verbose=verbose,
            no_init=no_init,
        )
        if not no_init:
            if opts.gen.deeplabv3.backbone == "resnet":
                assert Path(opts.gen.deeplabv3.pretrained_model.resnet).exists()

                std = torch.load(opts.gen.deeplabv3.pretrained_model.resnet)
                resnet.load_state_dict(
                    {
                        k.replace("backbone.", ""): v
                        for k, v in std.items()
                        if k.startswith("backbone.")
                    }
                )
                print("- Loaded pre-trained DeepLabv3+ Resnet101 Backbone as Encoder")
        return resnet

    elif opts.gen.deeplabv3.backbone == "mobilenet":
        assert Path(opts.gen.deeplabv3.pretrained_model.mobilenet).exists()
        mobilenet = MobileNetV2(
            no_init=no_init,
            pretrained_path=opts.gen.deeplabv3.pretrained_model.mobilenet,
        )
        print("- Loaded pre-trained DeepLabv3+ MobileNetV2 Backbone as Encoder")
        return mobilenet

    else:
        raise NotImplementedError("Unknown backbone in " + str(opts.gen.deeplabv3))


class DeeplabV2Encoder(nn.Module):
    def __init__(self, opts, no_init=False, verbose=0):
        """Deeplab architecture encoder
        """
        super().__init__()

        self.model = ResNetMulti(opts.gen.deeplabv2.nblocks, opts.gen.encoder.n_res)
        if opts.gen.deeplabv2.use_pretrained and not no_init:
            saved_state_dict = torch.load(opts.gen.deeplabv2.pretrained_model)
            new_params = self.model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split(".")
                if not i_parts[1] in ["layer5", "resblock"]:
                    new_params[".".join(i_parts[1:])] = saved_state_dict[i]
            self.model.load_state_dict(new_params)
            if verbose > 0:
                print("    - Loaded pretrained weights")

    def forward(self, x):
        return self.model(x)
