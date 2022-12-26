CUDA_VISIBLE_DEVICES=0 python train.py \
    --name cityscapes_own_cc_fpse_no_vae \
    --mpdist \
    --netG condconv \
    --dist_url tcp://localhost:8000 \
    --num_servers 1 \
    --netD fpse \
    --lambda_feat 20 \
    --dataset_mode cityscapes \
    --dataroot /media/group2/data/wanghaitao/cityscapes \
    --batchSize 1 \
    --niter 100 \
    --niter_decay 100 \
    --ngpus_per_node 1