_base_ = [
    './f1_q500_800x320.py',
]

num_frames_per_sample = 3
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            num_frames_per_sample=num_frames_per_sample,
        )
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
        score_threshold=0.2,
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

load_from = 'work_dirs/refactored/lidar/f1_q500_800x320-mini/iou/epoch_200.pth'
