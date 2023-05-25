from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ...utils.builder import BACKBONE
from ...utils.logger import logger
from ...utils.misc import enable_lower_param, param_size

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    https://github.com/facebookresearch/detectron2/blob/dfe8d368c8b7cc2be42c5c3faf9bdcc3c08257b1/detectron2/layers/batch_norm.py
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",

    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor):
        """
        initialized to perform identity transformation.
        The affine transform `x * weight + bias` will perform the equivalent
        computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
        """
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, version, block, layers, num_classes=1000, freeze_batchnorm=True, early_return=4):
        super(ResNet, self).__init__()

        self.name = f"resnet{version}"
        if freeze_batchnorm:
            self.bn_layer = FrozenBatchNorm2d
        else:
            self.bn_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.early_return = early_return
        assert self.early_return in [0, 1, 2, 3, 4], "wrong early return layer"

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.bn_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.features = 512 * block.expansion

        self.output_channel = self.inplanes

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, self.bn_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.bn_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.bn_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_pretrained(self, state_dict):
        logger.info(f"Loading {self.name} pretrained models")
        self.load_state_dict(state_dict)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, **kwargs) -> Dict:
        x = kwargs["image"]
        features = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features["res_layer1"] = x
        if self.early_return == 1:
            return features

        x = self.layer2(x)
        features["res_layer2"] = x
        if self.early_return == 2:
            return features

        x = self.layer3(x)
        features["res_layer3"] = x
        if self.early_return == 3:
            return features

        x = self.layer4(x)
        features["res_layer4"] = x

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        features["res_layer4_mean"] = x

        if self.early_return == 4:
            return features

        x = self.fc(x)
        out = {"res_output": x}
        out.update(features)
        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("18", BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("34", BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("50", Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("101", Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("152", Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls["resnet152"]))
    return model


def build_resnet(cfg):
    build_model = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }
    return build_model[cfg.TYPE](pretrained=cfg.PRETRAINED, freeze_batchnorm=cfg.FREEZE_BATCHNORM)


@BACKBONE.register_module()
class ResNet18(ResNet):

    def __init__(self, cfg):
        super(ResNet18, self).__init__(version="18",
                                       block=BasicBlock,
                                       layers=[2, 2, 2, 2],
                                       freeze_batchnorm=cfg.FREEZE_BATCHNORM,
                                       early_return=cfg.get("EARLY_RETURN", 4))
        if cfg.PRETRAINED:
            self.load_pretrained(model_zoo.load_url(model_urls["resnet18"]))


@BACKBONE.register_module()
class ResNet34(ResNet):

    def __init__(self, cfg):
        super(ResNet34, self).__init__(version="34",
                                       block=BasicBlock,
                                       layers=[3, 4, 6, 3],
                                       freeze_batchnorm=cfg.FREEZE_BATCHNORM,
                                       early_return=cfg.get("EARLY_RETURN", 4))
        if cfg.PRETRAINED:
            self.load_pretrained(model_zoo.load_url(model_urls["resnet34"]))


@BACKBONE.register_module()
class ResNet50(ResNet):

    def __init__(self, cfg):
        super(ResNet50, self).__init__(version="50",
                                       block=Bottleneck,
                                       layers=[3, 4, 6, 3],
                                       freeze_batchnorm=cfg.FREEZE_BATCHNORM,
                                       early_return=cfg.get("EARLY_RETURN", 4))
        if cfg.PRETRAINED:
            self.load_pretrained(model_zoo.load_url(model_urls["resnet50"]))


@BACKBONE.register_module()
class ResNet101(ResNet):

    def __init__(self, cfg):
        super(ResNet101, self).__init__(version="101",
                                        block=Bottleneck,
                                        layers=[3, 4, 23, 3],
                                        freeze_batchnorm=cfg.FREEZE_BATCHNORM,
                                        early_return=cfg.get("EARLY_RETURN", 4))
        if cfg.PRETRAINED:
            self.load_pretrained(model_zoo.load_url(model_urls["resnet101"]))


@BACKBONE.register_module()
class ResNet152(ResNet):

    def __init__(self, cfg):
        super(ResNet152, self).__init__(version="152",
                                        block=Bottleneck,
                                        layers=[3, 8, 36, 3],
                                        freeze_batchnorm=cfg.FREEZE_BATCHNORM,
                                        early_return=cfg.get("EARLY_RETURN", 4))
        if cfg.PRETRAINED:
            self.load_pretrained(model_zoo.load_url(model_urls["resnet152"]))
