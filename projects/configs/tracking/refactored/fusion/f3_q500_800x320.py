_base_ = "./f1_q500_800x320.py"

# Turn off rand_flip
ida_aug_conf = {
    "resize_lim": (0.47, 0.625),
    "final_dim": (320, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

train_pipeline_multiframe = [
    dict(type='TrackResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='Pack3DTrackInputs', 
         keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 
            'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'lidar2global', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats', 'timestamp', 'pad_shape'
        ])
]

test_pipeline_multiframe = [
    dict(type='TrackResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(
        type='Pack3DTrackInputs',
        keys=['img'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'timestamp', 'lidar2global',
            'pad_shape'
        ])
]

# change to 3 frames per sample
num_frames_per_sample = 3 # how many frames to train on
train_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
        pipeline_multiframe=train_pipeline_multiframe,
    ),
)
val_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
        pipeline_multiframe=test_pipeline_multiframe,
    ),
)
test_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
        pipeline_multiframe=test_pipeline_multiframe,
    ),
)

model = dict(
    tracking=True,
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
    bbox_coder=dict(
        max_num=150,
    )
)