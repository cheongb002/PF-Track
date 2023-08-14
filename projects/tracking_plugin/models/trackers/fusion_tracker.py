import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models import build_loss
from copy import deepcopy
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.dense_heads.petr_head import pos2posemb3d
from projects.tracking_plugin.core.instances import Instances
from .runtime_tracker import RunTimeTracker
from .spatial_temporal_reason import SpatialTemporalReasoner
from .utils import time_position_embedding, xyz_ego_transformation, normalize, denormalize

@DETECTORS.register_module()
class FusionTracker(Cam3DTracker):
    def extract_clip_feats(self, pts, img, img_metas):
        pass
    def extract_feat(self, pts, img, img_metas):
        # extract features from images
        # extract features from point clouds
        # fuse features
        pass
    def extract_pts_feat(self, pts):
        pass