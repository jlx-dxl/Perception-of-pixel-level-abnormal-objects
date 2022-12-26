_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/fs_lost_and_found.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
model = dict(pretrained='/root/autodl-tmp/pretrain/sem_fpn/fpn_r101_512x1024_80k_cityscapes_20200717_012416-c5800d4c.pth', backbone=dict(depth=101))