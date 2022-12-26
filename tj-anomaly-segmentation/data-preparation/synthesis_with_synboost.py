import os

from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
import sys

full_path = os.path.realpath(__file__)
root_path = os.path.join(os.path.dirname(full_path),'../')
sys.path.insert(0, root_path)
from options.test_options import TestOptions
# import network
# from optimizer import restore_snapshot
# from datasets import cityscapes
# from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()
opt.results_dir = '/root/autodl-tmp/data_dis/preprocess/1'
print('Starting Image Synthesis Process')

import sys

sys.path.insert(0, './image_synthesis')
import data
from models.pix2pix_model import Pix2PixModel

world_size = 1
rank = 0

# Corrects where dataset is in necesary format
opt.dataroot = os.path.join(opt.results_dir, 'temp')

opt.world_size = world_size
opt.gpu = 0
opt.mpdist = False
opt.checkpoints_dir = '/root/autodl-tmp/pretrain/models/'
# opt.no_instance = False
# opt.semantic_nc = 36

dataloader = data.create_dataloader(opt, world_size, rank)

model = Pix2PixModel(opt)
model.eval()

synthesis_fdr = os.path.join(opt.results_dir, 'synthesis_pixtopix')

if not os.path.exists(synthesis_fdr):
    os.makedirs(synthesis_fdr)
# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')

    image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))
    synthesis_final_img.save(os.path.join(synthesis_fdr, data_i['path'][0].rsplit("/")[-1]))

    # x_output = img_resize(image=x_sample_det[indice - start_index])['image']
    # image_tmp = Image.fromarray(np.uint8(x_output))
    # image_tmp.save(os.path.join(synthesis_fdr, dset.labels['relative_file_path_'][indice]))
    # img_path = data_i['path']
    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_label', data_i['label'][b]),
    #                            ('synthesized_image', generated[b])])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1], gray=True)