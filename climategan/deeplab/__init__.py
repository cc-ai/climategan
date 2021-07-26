from pathlib import Path

import torch
import torch.nn as nn
from climategan.deeplab.deeplab_v2 import DeepLabV2Decoder
from climategan.deeplab.deeplab_v3 import DeepLabV3Decoder
from climategan.deeplab.mobilenet_v3 import MobileNetV2
from climategan.deeplab.resnet101_v3 import ResNet101
from climategan.deeplab.resnetmulti_v2 import ResNetMulti


def create_encoder(opts, no_init=False, verbose=0):
    if opts.gen.encoder.architecture == "deeplabv2":
        if verbose > 0:
            print("  - Add Deeplabv2 Encoder")
        return DeeplabV2Encoder(opts, no_init, verbose)
    elif opts.gen.encoder.architecture == "deeplabv3":
        if verbose > 0:
            backone = opts.gen.deeplabv3.backbone
            print("  - Add Deeplabv3 ({}) Encoder".format(backone))
        return build_v3_backbone(opts, no_init)
    else:
        raise NotImplementedError(
            "Unknown encoder: {}".format(opts.gen.encoder.architecture)
        )


def create_segmentation_decoder(opts, no_init=False, verbose=0):
    if opts.gen.s.architecture == "deeplabv2":
        if verbose > 0:
            print("  - Add DeepLabV2Decoder")
        return DeepLabV2Decoder(opts)
    elif opts.gen.s.architecture == "deeplabv3":
        if verbose > 0:
            print("  - Add DeepLabV3Decoder")
        return DeepLabV3Decoder(opts, no_init)
    else:
        raise NotImplementedError(
            "Unknown Segmentation architecture: {}".format(opts.gen.s.architecture)
        )


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
                print(
                    "    - Loaded pre-trained DeepLabv3+ Resnet101 Backbone as Encoder"
                )
        return resnet

    elif opts.gen.deeplabv3.backbone == "mobilenet":
        assert Path(opts.gen.deeplabv3.pretrained_model.mobilenet).exists()
        mobilenet = MobileNetV2(
            no_init=no_init,
            pretrained_path=opts.gen.deeplabv3.pretrained_model.mobilenet,
        )
        print("    - Loaded pre-trained DeepLabv3+ MobileNetV2 Backbone as Encoder")
        return mobilenet

    else:
        raise NotImplementedError("Unknown backbone in " + str(opts.gen.deeplabv3))


class DeeplabV2Encoder(nn.Module):
    def __init__(self, opts, no_init=False, verbose=0):
        """Deeplab architecture encoder"""
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
