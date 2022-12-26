import torch

a = torch.load('/root/autodl-tmp/synthesis/logs/2022-05-05T12-47-10_cityscapes_scene_images_transformer/checkpoints/last.ckpt')
#a['global']
a['global_step'] = 109114
torch.save(a,'/root/autodl-tmp/synthesis/logs/2022-05-05T12-47-10_cityscapes_scene_images_transformer/checkpoints/last.ckpt')
#b = a['state_dict'].pop("cond_stage_model.colorize")