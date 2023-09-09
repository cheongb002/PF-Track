# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-50, -50.8, -5, 50, 49.2, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'bus', 'trailer', 
    'motorcycle', 'bicycle', 'pedestrian', 
    'construction_vehicle', 'traffic_cone', 'barrier' # non-detection classes must go at the end
]
metainfo = dict(classes=class_names, version='v1.0-trainval')
# metainfo = dict(classes=class_names, version='v1.0-mini')

dataset_type = 'NuScenesTrackingDataset'
data_root = './data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/nuscenes/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

# pkl paths
train_pkl_path = 'mmlab-v2/tracking_forecasting_infos_train.pkl'
test_pkl_path = 'mmlab-v2/tracking_forecasting_infos_val.pkl'
val_pkl_path = 'mmlab-v2/tracking_forecasting_infos_val.pkl'
# train_pkl_path = 'mmlab-v2/tracking_forecasting-mini_infos_val.pkl'
# test_pkl_path = 'mmlab-v2/tracking_forecasting-mini_infos_val.pkl'
# val_pkl_path = 'mmlab-v2/tracking_forecasting-mini_infos_val.pkl'

train_pipeline = [
    dict(
        type='mmdet3.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='mmdet3.LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='mmdet3.TrackLoadAnnotations3D', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_forecasting=True),
    dict(
        type='mmdet3.GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='mmdet3.BEVFusionRandomFlip3D'),
    dict(type='mmdet3.PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='mmdet3.TrackInstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='mmdet3.TrackObjectNameFilter', classes=class_names),
    dict(type='mmdet3.PointShuffle'),
]

train_pipeline_multiframe = [
    dict(type='mmdet3.Pack3DTrackInputs', 
         keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 
            'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'lidar2global', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats', 'timestamp', 'pad_shape'
        ])
]

test_pipeline = [
    dict(
        type='mmdet3.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='mmdet3.LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='mmdet3.PointsRangeFilter',
        point_cloud_range=point_cloud_range),
]

test_pipeline_multiframe = [
    dict(
        type='mmdet3.Pack3DTrackInputs',
        keys=['points'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'timestamp', 'lidar2global',
            'pad_shape'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            num_frames_per_sample=1, # default single frame training
            forecasting=True,
            data_root=data_root,
            ann_file=train_pkl_path,
            pipeline=train_pipeline,
            pipeline_multiframe=train_pipeline_multiframe,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            use_valid_flag=True,
            box_type_3d='LiDAR',
            filter_empty_gt=True,
            backend_args=backend_args
        )
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        num_frames_per_sample=1,
        data_root=data_root,
        ann_file=test_pkl_path,
        pipeline=test_pipeline,
        pipeline_multiframe=test_pipeline_multiframe,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        num_frames_per_sample=1,
        data_root=data_root,
        ann_file=val_pkl_path,
        pipeline=test_pipeline,
        pipeline_multiframe=test_pipeline_multiframe,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        backend_args=backend_args))

val_evaluator = dict(
    # type='NuScenesTrackingMetric',
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + val_pkl_path,
    metric='bbox',
    jsonfile_prefix='work_dirs/nuscenes_results/detection',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
