# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from copy import deepcopy

import numpy as np
import torch
from mmdet3d.registry import MODELS
from mmengine.structures import InstanceData

from projects.BEVFusion.bevfusion.ops import Voxelization
from projects.tracking_plugin.core.instances import Instances

from .Cam3DTracker import Cam3DTracker, track_bbox3d2result


@MODELS.register_module()
class Fusion3DTracker(Cam3DTracker):
    def __init__(
            self, 
            *args, 
            view_transform:dict, 
            voxelize_cfg:dict, 
            pts_voxel_encoder:dict,
            pts_middle_encoder:dict,
            pts_backbone:dict,
            pts_neck:dict,
            fusion_layer:dict,
            batch_clip:bool=True, 
            **kwargs
        ):
        """Fusion3DTracker.
        Args:
            batch_clip (bool, optional): Whether to put frames from clip into a single
            batch. If out of memory errors, set to False.
        """
        super().__init__(*args, **kwargs)
        self.batch_clip = batch_clip
        self.view_transform = MODELS.build(view_transform)
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

    def loss(
            self,
            inputs:dict,
            data_samples):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                For each sample, the format is Dict(key: contents of the timestamps)
                Defaults to None. For each field, its shape is [T * NumCam * ContentLength]
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None. Number same as batch size.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, T, Num_Cam, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
            l2g_r_mat (list[Tensor]). Lidar to global transformation, shape [T, 3, 3]
            l2g_t (list[Tensor]). Lidar to global rotation
                points @ R_Mat + T
            timestamp (list). Timestamp of the frames
        Returns:
            dict: Losses of different branches.
        """
        img = inputs['imgs']
        num_frame = len(img)
        batch_size = img[0].shape[0]
        assert batch_size == 1, "Currently only support batch size 1"

        # extract the features of multi-frame images and points
        fused_feats = self.extract_clip_feats(inputs, data_samples)

        # Empty the runtime_tracker
        # Use PETR head to decode the bounding boxes on every frame
        outs = list()
        # returns only single set of track instances, corresponding to single batch
        next_frame_track_instances = self.generate_empty_instance()
        device = next_frame_track_instances.reference_points.device

        # Running over all the frames one by one
        self.runtime_tracker.empty()
        for frame_idx, data_samples_frame in enumerate(data_samples):
            img_metas_single_frame = [ds_batch.metainfo for ds_batch in data_samples_frame]
            ff_gt_bboxes_list = [ds_batch.gt_instances_3d.bboxes_3d for ds_batch in data_samples_frame]
            ff_gt_labels_list = [ds_batch.gt_instances_3d.labels_3d for ds_batch in data_samples_frame]
            ff_instance_inds = [ds_batch.gt_instances_3d.instance_inds for ds_batch in data_samples_frame]
            gt_forecasting_locs = data_samples_frame[0].gt_instances_3d.forecasting_locs
            gt_forecasting_masks = data_samples_frame[0].gt_instances_3d.forecasting_masks

            # PETR detection head
            track_instances = next_frame_track_instances
            out = self.pts_bbox_head(
                fused_feats[frame_idx], img_metas_single_frame, 
                track_instances.query_feats, track_instances.query_embeds, 
                track_instances.reference_points)

            # 1. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 2. Loss computation for the detection
            out['loss_dict'] = self.criterion.loss_single_frame(
                frame_idx, ff_gt_bboxes_list, ff_gt_labels_list,
                ff_instance_inds, out, None)

            # 3. Spatial-temporal reasoning
            track_instances = self.STReasoner(track_instances)

            if self.STReasoner.history_reasoning:
                out['loss_dict'] = self.criterion.loss_mem_bank(
                    frame_idx,
                    out['loss_dict'],
                    ff_gt_bboxes_list,
                    ff_gt_labels_list,
                    ff_instance_inds,
                    track_instances)
            
            if self.STReasoner.future_reasoning:
                active_mask = (track_instances.obj_idxes >= 0)
                out['loss_dict'] = self.forward_loss_prediction(
                    frame_idx,
                    out['loss_dict'],
                    track_instances[active_mask],
                    gt_forecasting_locs,
                    gt_forecasting_masks,
                    ff_instance_inds)

            # 4. Prepare for next frame
            track_instances = self.frame_summarization(track_instances, tracking=False)
            active_mask = self.runtime_tracker.get_active_mask(track_instances, training=True)
            track_instances.track_query_mask[active_mask] = True
            active_track_instances = track_instances[active_mask]
            if self.motion_prediction and frame_idx < num_frame - 1:
                # assume batch size is 1
                time_delta = data_samples[frame_idx+1][0].metainfo['timestamp'] - data_samples[frame_idx][0].metainfo['timestamp']
                active_track_instances = self.update_reference_points(
                    active_track_instances,
                    time_delta,
                    use_prediction=self.motion_prediction_ref_update,
                    tracking=False)
            if self.if_update_ego and frame_idx < num_frame - 1:
                active_track_instances = self.update_ego(
                    active_track_instances, 
                    data_samples[frame_idx][0].metainfo['lidar2global'].to(device), 
                    data_samples[frame_idx + 1][0].metainfo['lidar2global'].to(device),
                )
            if frame_idx < num_frame - 1:
                active_track_instances = self.STReasoner.sync_pos_embedding(active_track_instances, self.query_embedding)
            
            empty_track_instances = self.generate_empty_instance()
            next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])
            self.runtime_tracker.frame_index += 1
            outs.append(out)
        losses = self.criterion(outs)
        self.runtime_tracker.empty()
        return losses

    def predict(self, inputs:dict, data_samples):
        imgs = inputs['imgs'] # bs, num frames, num cameras (6), C, H, W
        batch_size = len(imgs)
        assert batch_size == 1, "Only support single bs prediction"
        num_frame = imgs[0].shape[0]
        assert num_frame == 1, "Only support single frame prediction"

        # extract the features of multi-frame images and points
        fused_feats = self.extract_clip_feats(inputs, data_samples)
        
        # new sequence
        timestamp = data_samples[0][0].metainfo['timestamp']
        if self.runtime_tracker.timestamp is None or abs(timestamp - self.runtime_tracker.timestamp) > 10:
            self.runtime_tracker.timestamp = timestamp
            self.runtime_tracker.current_seq += 1
            self.runtime_tracker.track_instances = None
            self.runtime_tracker.current_id = 0
            self.runtime_tracker.l2g = None
            self.runtime_tracker.time_delta = 0
            self.runtime_tracker.frame_index = 0
        self.runtime_tracker.time_delta = timestamp - self.runtime_tracker.timestamp
        self.runtime_tracker.timestamp = timestamp
        
        # processing the queries from t-1
        prev_active_track_instances = self.runtime_tracker.track_instances
        for frame_idx in range(num_frame): # TODO remove this for loop, assume num_frame = 1 for prediction
            img_metas_single_frame = [ds[frame_idx].metainfo for ds in data_samples]
        
            # 1. Update the information of previous active tracks
            if prev_active_track_instances is None:
                track_instances = self.generate_empty_instance()
            else:
                device = prev_active_track_instances.reference_points.device
                if self.motion_prediction:
                    time_delta = self.runtime_tracker.time_delta
                    prev_active_track_instances = self.update_reference_points(
                        prev_active_track_instances, time_delta, 
                        use_prediction=self.motion_prediction_ref_update, tracking=True)
                if self.if_update_ego:
                    prev_active_track_instances = self.update_ego(
                        prev_active_track_instances, self.runtime_tracker.l2g.to(device), 
                        img_metas_single_frame[0]['lidar2global'].to(device))
                prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
                track_instances = Instances.cat([self.generate_empty_instance(), prev_active_track_instances])

            self.runtime_tracker.l2g = img_metas_single_frame[0]['lidar2global']
            self.runtime_tracker.timestamp = img_metas_single_frame[0]['timestamp']

            # 2. PETR detection head
            out = self.pts_bbox_head(
                fused_feats[frame_idx], img_metas_single_frame, track_instances.query_feats,
                track_instances.query_embeds, track_instances.reference_points)
            # 3. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 4. Spatial-temporal Reasoning
            self.STReasoner(track_instances)
            track_instances = self.frame_summarization(track_instances, tracking=True)
            out['all_cls_scores'][-1][0, :] = track_instances.logits
            out['all_bbox_preds'][-1][0, :] = track_instances.bboxes

            if self.STReasoner.future_reasoning:
                # motion forecasting has the shape of [num_query, T, 2]
                out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
            else:
                out['all_motion_forecasting'] = None

            # 5. Track class filtering: before decoding bboxes, only leave the objects under tracking categories
            if self.tracking:
                max_cat = torch.argmax(out['all_cls_scores'][-1, 0, :].sigmoid(), dim=-1)
                related_cat_mask = (max_cat < 7) # we set the first 7 classes as the tracking classes of nuscenes
                track_instances = track_instances[related_cat_mask]
                out['all_cls_scores'] = out['all_cls_scores'][:, :, related_cat_mask, :]
                out['all_bbox_preds'] = out['all_bbox_preds'][:, :, related_cat_mask, :]
                if out['all_motion_forecasting'] is not None:
                    out['all_motion_forecasting'] = out['all_motion_forecasting'][related_cat_mask, ...]

                # 6. assign ids
                active_mask = (track_instances.scores > self.runtime_tracker.threshold)
                for i in range(len(track_instances)):
                    if track_instances.obj_idxes[i] < 0:
                        track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
                        self.runtime_tracker.current_id += 1
                        if active_mask[i]:
                            track_instances.track_query_mask[i] = True
                out['track_instances'] = track_instances

                # 7. Prepare for the next frame and output
                score_mask = (track_instances.scores > self.runtime_tracker.output_threshold)
                out['all_masks'] = score_mask.clone()
                self.runtime_tracker.update_active_tracks(track_instances, active_mask)

            bbox_list = self.pts_bbox_head.get_bboxes(out, img_metas_single_frame, tracking=True)
            # self.runtime_tracker.update_active_tracks(active_track_instances)

            # each time, only run one frame
            self.runtime_tracker.frame_index += 1
            break

        bbox_results = [
            track_bbox3d2result(bboxes, scores, labels, obj_idxes, track_scores, forecasting)
            for bboxes, scores, labels, obj_idxes, track_scores, forecasting in bbox_list
        ]
        if self.tracking:
            bbox_results[0]['track_ids'] = [f'{self.runtime_tracker.current_seq}-{i}' for i in bbox_results[0]['track_ids'].long().cpu().numpy().tolist()]

        results_list_3d = [InstanceData(metainfo=results) for results in bbox_results]
        detsamples = self.add_pred_to_datasample(
            data_samples[0], data_instances_3d=results_list_3d
        )
        return detsamples


    def extract_clip_feats(self, inputs, data_samples):
        outputs = list()
        img_clip = inputs['imgs']
        pts_clip = inputs['points']
        num_frames = len(img_clip)
        batch_size, num_came = img_clip[0].shape[0:2]
        if self.batch_clip:
            # put batched frames from clip into a single superbatch
            # single frame image, N * NumCam * C * H * W
            imgs_stacked = torch.cat([frame for frame in img_clip], dim=0)
            pts_stacked = []
            input_metas = []
            for pts_i, data_sample_i in zip(pts_clip, data_samples):
                pts_stacked.extend(pts_i)
                input_metas.extend([ds_batch.metainfo for ds_batch in data_sample_i])
            # extract features from superbatch
            input_dict = dict(img=imgs_stacked, pts=pts_stacked)
            fused_feats = self.extract_feat(input_dict, input_metas)
            # extract output from superbatch back to clip
            for frame_idx in range(num_frames):
                outputs.append(fused_feats[frame_idx * batch_size:(frame_idx + 1) * batch_size])
        else:
            # process each frame in clip separately
            for frame_idx, (img_frame, pts_frame) in enumerate(zip(img_clip, pts_clip)):
                input_dict = dict(img=img_frame, pts=pts_frame)
                # iterate over each item in batch for given frame_idx
                input_metas = [item[frame_idx].metainfo for item in data_samples]
                fused_feats = self.extract_feat(input_dict, input_metas)
                outputs.append(fused_feats)
        return outputs

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.pts_fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x
    
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes