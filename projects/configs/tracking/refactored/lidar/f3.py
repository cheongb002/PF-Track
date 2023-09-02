_base_ = [
    './f1_q500_800x320.py',
]

# turn off cbgs to speed up training
num_frames_per_sample = 3
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        num_frames_per_sample=num_frames_per_sample,
        forecasting=True,
        data_root={{_base_.data_root}},
        ann_file={{_base_.train_pkl_path}},
        pipeline={{_base_.train_pipeline}},
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

model = dict(
    tracking=True,
    train_backbone=False,
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

val_evaluator = dict(type='NuScenesTrackingMetric')
test_evaluator = dict(type='NuScenesTrackingMetric')
load_from = 'ckpts/f1_heatmap/epoch_20.pth'
