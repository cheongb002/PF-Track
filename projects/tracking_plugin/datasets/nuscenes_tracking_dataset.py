# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import List, Union

import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
from mmengine.dataset import Compose
from pyquaternion import Quaternion


@DATASETS.register_module()
class NuScenesTrackingDataset(NuScenesDataset):
    CLASSES = ('car', 'truck', 'bus', 'trailer', 
               'motorcycle', 'bicycle', 'pedestrian', 
               'construction_vehicle', 'traffic_cone', 'barrier')
    def __init__(self,
                 pipeline_multiframe=None,
                 num_frames_per_sample=2,
                 forecasting=False,
                 ratio=1,
                 *args, **kwargs,
                 ):
        self.num_frames_per_sample = num_frames_per_sample
        self.pipeline_multiframe = pipeline_multiframe
        self.forecasting = forecasting
        self.ratio = ratio # divide the samples by a certain ratio, useful for quicker ablations
        if self.pipeline_multiframe is not None:
            self.pipeline_multiframe = Compose(self.pipeline_multiframe)
        self.scene_tokens = []
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        if not self.test_mode:
            return super().__len__() // self.ratio
        else:
            return super().__len__()

    def prepare_data(self, index: int) -> dict | None:
        input_dict = super().prepare_data(index)
        if input_dict is None:
            return None
        ann_info = input_dict['ann_info'] if not self.test_mode \
            else input_dict['eval_ann_info']
        if self.filter_empty_gt and \
                (input_dict is None or ~(ann_info['gt_labels_3d'] != -1).any()):
            return None
        scene_token = input_dict['scene_token']
        data_queue = [input_dict]

        index_list = self.generate_track_data_indexes(index)
        index_list = index_list[::-1]
        for i in index_list[1:]:
            data_info_i = super().prepare_data(i)
            if data_info_i is None or data_info_i['scene_token'] != scene_token:
                return None
            ann_info = data_info_i['ann_info'] if not self.test_mode \
                else data_info_i['eval_ann_info']
            if self.filter_empty_gt and \
                (data_info_i is None or
                    ~(ann_info['gt_labels_3d'] != -1).any()):
                return None
            data_queue.append(data_info_i)

        # return to the normal frame order
        data_queue = data_queue[::-1]

        # construct dict of lists
        sample_data = dict()
        for key in data_queue[-1].keys():
            sample_data[key] = list()
        for d in data_queue:
            for k, v in d.items():
                sample_data[k].append(v)

        # multiframe processing
        data = self.pipeline_multiframe(sample_data)
        return data

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        data_info = super().parse_data_info(info)
        self.scene_tokens.append(data_info['scene_token'])
        # ego movement represented by lidar2global
        l2e = np.array(data_info['lidar_points']['lidar2ego'])
        e2g = np.array(data_info['ego2global'])
        l2g = e2g @ l2e

        # points @ R.T + T
        data_info.update(lidar2global=l2g.astype(np.float32))

        if self.modality['use_camera']:
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['images'].items():
                # obtain lidar to image transformation matrix
                lidar2cam_rt = np.array(cam_info['lidar2cam'])
                intrinsic = np.array(cam_info['cam2img'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt) # the transpose of extrinsic matrix

            data_info.update(
                dict(
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                ))
        return data_info

    def generate_track_data_indexes(self, index):
        """Choose the track indexes that are within the same sequence
        """
        index_list = [i for i in range(index - self.num_frames_per_sample + 1, index + 1)]
        scene_tokens = [self.scene_tokens[i] for i in index_list]
        tgt_scene_token, earliest_index = scene_tokens[-1], index_list[-1]
        for i in range(self.num_frames_per_sample)[::-1]:
            if scene_tokens[i] == tgt_scene_token:
                earliest_index = index_list[i]
            elif self.test_mode:
                index_list = index_list[i + 1:]
                break
            elif (not self.test_mode):
                index_list[i] = earliest_index
        return index_list
