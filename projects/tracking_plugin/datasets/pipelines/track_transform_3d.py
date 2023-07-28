# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
from mmdet3d.datasets.transforms.transforms_3d import (ObjectNameFilter,
                                                       ObjectRangeFilter)
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                LiDARInstance3DBoxes)
from projects.PETR.petr.transforms_3d import ResizeCropFlipImage, GlobalRotScaleTransImage
from PIL import Image


@TRANSFORMS.register_module()
class TrackInstanceRangeFilter(ObjectRangeFilter):

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]
        else:
            raise TypeError(
                f'Invalid points instance type '
                f'{type(input_dict["gt_bboxes_3d"])}')

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        instance_inds = input_dict['instance_inds']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        instance_inds = instance_inds[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['instance_inds'] = instance_inds

        # hacks for forecasting
        if 'gt_forecasting_locs' in input_dict.keys():
            input_dict['gt_forecasting_locs'] = input_dict['gt_forecasting_locs'][mask.numpy().astype(bool)]
            input_dict['gt_forecasting_masks'] = input_dict['gt_forecasting_masks'][mask.numpy().astype(bool)]
            input_dict['gt_forecasting_types'] = input_dict['gt_forecasting_types'][mask.numpy().astype(bool)]

        return input_dict

@TRANSFORMS.register_module()
class TrackObjectNameFilter(ObjectNameFilter):
    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=bool)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['instance_inds'] = input_dict['instance_inds'][gt_bboxes_mask]

        # hacks for forecasting
        if 'gt_forecasting_locs' in input_dict.keys():
            input_dict['gt_forecasting_locs'] = input_dict['gt_forecasting_locs'][gt_bboxes_mask]
            input_dict['gt_forecasting_masks'] = input_dict['gt_forecasting_masks'][gt_bboxes_mask]
            input_dict['gt_forecasting_types'] = input_dict['gt_forecasting_types'][gt_bboxes_mask]
        
        return input_dict

@TRANSFORMS.register_module()
class TrackResizeCropFlipImage(ResizeCropFlipImage):
    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        frame_num = len(results['timestamp'])
        imgs = results["img"]
        N = len(imgs[0])
        new_imgs = [[] for _ in range(frame_num)]
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for t in range(frame_num):
            for i in range(N):
                img = Image.fromarray(np.uint8(imgs[t][i]))
                # augmentation (resize, crop, horizontal flip, rotate)
                img, ida_mat = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_imgs[t].append(np.array(img).astype(np.float32))
                results['intrinsics'][t][i][:3, :3] = ida_mat @ results['intrinsics'][t][i][:3, :3]

        results["img"] = new_imgs
        for t in range(frame_num):
            results['lidar2img'][t] = [results['intrinsics'][t][i] @ results['extrinsics'][t][i].T for i in range(len(results['extrinsics'][t]))]

        return results


@TRANSFORMS.register_module()
class TrackGlobalRotScaleTransImage(GlobalRotScaleTransImage):

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline. Containing multiple frames.
        Returns:
            dict: Updated result dict.
        """
        frame_num = len(results['timestamp'])

        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle, frame_num)
        if self.reverse_angle:
            rot_angle *= -1
        
        for i in range(frame_num):
            results["gt_bboxes_3d"][i].rotate(
                np.array(rot_angle)
            )  

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio, frame_num)
        for i in range(frame_num):
            results["gt_bboxes_3d"][i].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle, frame_num):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for i in range(frame_num):
            for view in range(num_view):
                results["lidar2img"][i][view] = (torch.tensor(results["lidar2img"][i][view]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio, frame_num):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for i in range(frame_num):
            for view in range(num_view):
                results["lidar2img"][i][view] = (torch.tensor(results["lidar2img"][i][view]).float() @ rot_mat_inv).numpy()

        return
