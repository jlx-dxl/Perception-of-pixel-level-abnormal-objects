#!/usr/bin/env bash

    # Example on Mapillary
     CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 ../train.py \
        --dataset mapillary \
        --arch network.deepv3.DeepSRNX50V3PlusD_m1 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --syncbn \
        --sgd \
        --lr 2e-2 \
        --lr_schedule poly \
        --poly_exp 1.0 \
        --crop_size 896 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --gblur \
        --max_epoch 3 \
        --img_wt_loss \
        --wt_bound 6.0 \
        --bs_mult 2 \
        --apex \
        --exp mapillary_pretrain \
        --ckpt ./logs/ \
        --tb_path ./logs/
