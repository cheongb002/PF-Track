_base_ = [
    '../../../_base_/datasets/nus-tracking-3d-lidar.py',
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/cosine.py',
    '../../../_base_/models/pftrack2-lidar.py'
]
custom_imports = dict(imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)
custom_imports = dict(imports=['projects.tracking_plugin'], allow_failed_imports=False)


point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model = dict(
    pc_range = point_cloud_range,
    spatial_temporal_reason=dict(pc_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range)),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            assigner=dict(pc_range=point_cloud_range)
        ),
    ),
    loss=dict(assigner=dict(pc_range=point_cloud_range)),
)

workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/'

file_client_args = dict(backend='disk')

num_frames_per_sample = 1 # how many frames to train on
train_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
    )
)
test_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
    )
)
val_dataloader = dict(
    dataset=dict(
        num_frames_per_sample=num_frames_per_sample,
    )
)

# optimizer
lr = 2e-4  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.01, betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1)))
)

num_epochs = 12
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=num_epochs,
        end=num_epochs,
        by_epoch=True,
        eta_min=1e-7)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=1)
find_unused_parameters=False

# load_from='ckpts/f1/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from=None
default_hooks=dict(
    logger=dict(interval=1)
)