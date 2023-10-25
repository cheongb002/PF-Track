_base_ = [
    '../../../_base_/datasets/nus-tracking-3d-lidar.py',
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/cyclic-20e.py',
    '../../../_base_/models/pftrack2-lidar.py'
]
custom_imports = dict(imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)
custom_imports = dict(imports=['projects.tracking_plugin'], allow_failed_imports=False)


point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] # from nuscenes config
model = dict(
    train_backbone=False,
    voxelize_cfg=dict(point_cloud_range=point_cloud_range),
    pc_range = point_cloud_range,
    spatial_temporal_reason=dict(pc_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range)),
    test_cfg=dict(
        point_cloud_range=point_cloud_range,
        dataset='nuScenes'
    ),
    loss=dict(assigner=dict(pc_range=point_cloud_range)),
)

workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/'

file_client_args = dict(backend='disk')

num_epochs = 20
lr = 5e-5
lr_max = 5e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.01, betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(
        img_backbone=dict(lr_mult=0.1),
        # pts_voxel_encoder=dict(lr_mult=0.1),
        # pts_middle_encoder=dict(lr_mult=0.1),
        # pts_backbone=dict(lr_mult=0.1),
        # pts_neck=dict(lr_mult=0.1),
        ))
)

# for LR tuning
# param_scheduler = dict(
#     _delete_=True,
#     type='PolyLR',
#     eta_min=2e-3,
#     power=0.9,
#     by_epoch=False,
# )

param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from lr to lr_max
    # during the next 12 epochs, learning rate decreases from lr_max to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=int(0.4*num_epochs),
        eta_min=lr_max,
        begin=0,
        end=int(0.4*num_epochs),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs-int(0.4*num_epochs),
        eta_min=lr * 1e-4,
        begin=int(0.4*num_epochs),
        end=num_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum decreases from the 0.95 to 0.85
    # during the next 12 epochs, momentum increases from 0.85 to 0.95
    dict(
        type='CosineAnnealingMomentum',
        T_max=int(0.4*num_epochs),
        eta_min=0.85,
        begin=0,
        end=int(0.4*num_epochs),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=num_epochs-int(0.4*num_epochs),
        eta_min=0.95,
        begin=int(0.4*num_epochs),
        end=num_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(max_epochs=num_epochs, val_interval=4)
find_unused_parameters=True

load_from='ckpts/BEVFusion/lidar/epoch_20.pth'
resume_from=None
default_hooks=dict(
    logger=dict(interval=500), 
    checkpoint=dict(interval=1)
)

randomness = dict(seed=1792775515)