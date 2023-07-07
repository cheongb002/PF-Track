# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
__all__ = ['VoVNet', 'VoVNetCP']

from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt

from .resnet import *
from .second import *
from .sparse_encoder import *
from .pillar_encoder import *
from .dla import *