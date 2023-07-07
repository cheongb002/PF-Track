from typing import Any, Dict

from abc import ABCMeta
from collections import OrderedDict

from torch import nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)

__all__ = ["BEVFusionBackbone"]


class BEVFusionBackbone(BaseModule, metaclass=ABCMeta):
    def __init__(self, encoders: Dict[str, Any], fuser: Dict[str, Any]) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )