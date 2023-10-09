_base_ = [
    './f1_q500_800x320.py',
]

# remove flip and rot/scale/trans augmentations
train_pipeline = [
    dict(
        type='mmdet3d.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args={{_base_.backend_args}}),
    dict(
        type='mmdet3d.LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args={{_base_.backend_args}}),
    dict(type='mmdet3d.TrackLoadAnnotations3D', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_forecasting=True),
    dict(type='mmdet3d.PointsRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(type='mmdet3d.TrackInstanceRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(type='mmdet3d.TrackObjectNameFilter', classes={{_base_.class_names}}),
    dict(type='mmdet3d.PointShuffle'),
]

# turn off cbgs to speed up training
num_frames_per_sample = 3
# without CBGS
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        num_frames_per_sample=num_frames_per_sample,
        forecasting=True,
        data_root={{_base_.data_root}},
        ann_file={{_base_.train_pkl_path}},
        pipeline=train_pipeline,
        pipeline_multiframe={{_base_.train_pipeline_multiframe}},
        metainfo={{_base_.metainfo}},
        modality={{_base_.input_modality}},
        test_mode=False,
        data_prefix={{_base_.data_prefix}},
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        use_valid_flag=True,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        backend_args={{_base_.backend_args}}
    )
)

# With CBGS
# train_dataloader = dict(
#     dataset=dict(
#         dataset=dict(
#             num_frames_per_sample=num_frames_per_sample,
#             pipeline=train_pipeline,
#         )
#     )
# )

model = dict(
    tracking=True,
    train_backbone=True,
    if_update_ego=True,
    motion_prediction=True,
    motion_prediction_ref_update=True,
    runtime_tracker=dict(
        output_threshold=0.2,
        score_threshold=0.4,
        record_threshold=0.4,
        max_age_since_update=7,
    ),
    spatial_temporal_reason=dict(
        history_reasoning=True,
        future_reasoning=True,
        fut_len=8,
    ),
    pts_bbox_head=dict(
        max_num=150
    ),
)

val_evaluator = dict(type='NuScenesTrackingMetric', jsonfile_prefix='work_dirs/nuscenes_results/tracking')
test_evaluator = dict(type='NuScenesTrackingMetric', jsonfile_prefix='work_dirs/nuscenes_results/tracking')
load_from = 'ckpts/f1_heatmap/epoch_20.pth'
