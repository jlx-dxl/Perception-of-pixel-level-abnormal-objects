# the whole file use the model trained segmentation and resynthesis modeL to get the
# segmentation result and resynthesis result

import os

from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

from options.test_options import TestOptions
import sys
sys.path.insert(0, './image_segmentation') #the working dir is changed to the ./image_segmentation
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()

opt.snapshot = "/root/autodl-tmp/pretrain/models/image-segmentation/cityscapes_best.pth"
opt.demo_folder = "/root/autodl-tmp/data_dis/preprocess/fs_lost_and_found_val/original"
opt.results_dir = "/root/autodl-tmp/data_dis/preprocess/fs_lost_and_found_val"
opt.no_segmentation = False

# if not opt.no_segmentation: #in this branch, we need segmentations
#     assert_and_infer_cfg(opt, train_mode=False)
#     cudnn.benchmark = False
#     torch.cuda.empty_cache()
#
#     # Get segmentation Net
#     opt.dataset_cls = cityscapes
#     net = network.get_net(opt, criterion=None) # to do : this is in the ./image_segmentation/network/_init_.py
#     net = torch.nn.DataParallel(net).cuda()
#     print('Segmentation Net built.')
    # # load snapshot from opt.snapshot with net and optimizer, it can ignore wrong size parameter
    # net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
    # net.eval()
    # print('Segmentation Net Restored.')
    #
    # # Get RGB Original Images
    # data_dir = opt.demo_folder #load sample image from demo folder
    # images = os.listdir(data_dir)
    # if len(images) == 0:
    #     print('There are no images at directory %s. Check the data path.' % (data_dir))
    # else:
    #     print('There are %d images to be processed.' % (len(images)))
    # images.sort()
    #
    # # Transform images to Tensor based on ImageNet Mean and STD
    # mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    #
    # # Create save directory
    # if not os.path.exists(opt.results_dir):
    #     os.makedirs(opt.results_dir)
    #
    # color_mask_fdr = os.path.join(opt.results_dir, 'color-mask')
    # overlap_fdr = os.path.join(opt.results_dir, 'overlap')
    # semantic_label_fdr = os.path.join(opt.results_dir, 'semantic_labelIds')
    # semantic_fdr = os.path.join(opt.results_dir, 'semantic')
    # soft_fdr = os.path.join(opt.results_dir, 'entropy')
    # soft_fdr_2 = os.path.join(opt.results_dir, 'logit_distance')
    # # original_fdr = os.path.join(opt.results_dir, 'original')
    #
    # if not os.path.exists(color_mask_fdr):
    #     os.makedirs(color_mask_fdr)
    #
    # if not os.path.exists(overlap_fdr):
    #     os.makedirs(overlap_fdr)
    #
    # if not os.path.exists(semantic_fdr):
    #     os.makedirs(semantic_fdr)
    #
    # if not os.path.exists(semantic_label_fdr):
    #     os.makedirs(semantic_label_fdr)
    #
    # if not os.path.exists(soft_fdr):
    #     os.makedirs(soft_fdr)
    #
    # if not os.path.exists(soft_fdr_2):
    #     os.makedirs(soft_fdr_2)
    #
    # # if not os.path.exists(original_fdr):
    #     # os.makedirs(original_fdr)
    #
    # # creates temporary folder to adapt format to image synthesis which use cityscapes datasets
    # if not os.path.exists(os.path.join(opt.results_dir, 'temp')):
    #     os.makedirs(os.path.join(opt.results_dir, 'temp'))
    #     os.makedirs(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val'))
    #     os.makedirs(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val'))
    #
    # # Loop around all figures
    # for img_id, img_name in enumerate(images):
    #     img_dir = os.path.join(data_dir, img_name)
    #     img = Image.open(img_dir).convert('RGB')
    #     # img.save(os.path.join(original_fdr, img_name))
    #     # img.save(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val', img_name[:-4] + '_leftImg8bit.png'))
    #     img_tensor = img_transform(img)
    #
    #     # predict
    #     with torch.no_grad():
    #         pred = net(img_tensor.unsqueeze(0).cuda()) # unsqueeze to have the batch dimention
    #         print('%04d/%04d: Segmentation Inference done.' % (img_id + 1, len(images)))
    #
    #     softmax = torch.nn.functional.softmax(torch.tensor(pred), dim=1)
    #     pred = pred.cpu().numpy().squeeze()
    #     softmax_pred = torch.sum(-softmax*torch.log(softmax), dim=1)
    #     softmax_pred = (softmax_pred - softmax_pred.min()) / softmax_pred.max()
    #
    #     # get logit distance
    #     distance, _ = torch.topk(softmax, 2, dim=1)
    #     max_logit = distance[:, 0, :, :]
    #     max2nd_logit = distance[:, 1, :, :]
    #     result = max_logit - max2nd_logit
    #     map_logit = 1 - (result - result.min()) / result.max()
    #
    #     softmax_pred_og = softmax_pred.cpu().numpy().squeeze()
    #     map_logit = map_logit.cpu().numpy().squeeze()
    #     softmax_pred_og = softmax_pred_og * 255
    #     map_logit = map_logit * 255
    #     pred_name = 'entropy_' + img_name.rsplit('/')[-1]
    #     pred_name_2 = 'distance_' + img_name.rsplit('/')[-1]
    #     cv2.imwrite(os.path.join(soft_fdr, pred_name), softmax_pred_og)
    #     cv2.imwrite(os.path.join(soft_fdr_2, pred_name_2), map_logit)
    #
    #     pred = np.argmax(pred, axis=0) # get the index of the biggest probability class
    #
    #     color_name = 'color_mask_' + img_name
    #     overlap_name = 'overlap_' + img_name
    #     pred_name = 'pred_mask_' + img_name
    #
    #     # save colorized predictions
    #     colorized = opt.dataset_cls.colorize_mask(pred)
    #     #colorized.save(os.path.join(color_mask_fdr, color_name))
    #
    #     # save colorized predictions overlapped on original images
    #     overlap = cv2.addWeighted(np.array(img), 0.5, np.array(colorized.convert('RGB')), 0.5, 0)
    #     cv2.imwrite(os.path.join(overlap_fdr, overlap_name), overlap[:, :, ::-1])
    #
    #     # save label-based predictions, e.g. for submission purpose
    #     # label_out = pred # because we choose to predict whole 34 labels
    #     label_out = np.zeros_like(pred)
    #     for label_id, train_id in opt.dataset_cls.id_to_trainid.items():
    #         label_out[np.where(pred == train_id)] = label_id
    #     cv2.imwrite(os.path.join(semantic_label_fdr, pred_name), label_out)
    #     cv2.imwrite(os.path.join(semantic_fdr, pred_name), pred)
    #     cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'), label_out)
    #     cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'), label_out)
    #
    # print('Segmentation Results saved.')

# print('Starting Image Synthesis Process')
#
# import sys
# sys.path.insert(0, './image_synthesis')
# import data
# from models.pix2pix_model import Pix2PixModel
# from util.visualizer import Visualizer
# from util import html
#
# world_size = 1
# rank = 0
#
# # Corrects where dataset is in necesary format
# opt.dataroot = os.path.join(opt.results_dir, 'temp')
#
# opt.world_size = world_size
# opt.gpu = 0
# opt.mpdist = False #not use distributed multiprocessing
# opt.checkpoints_dir = '/root/autodl-tmp/pretrain/models/'
#
# dataloader = data.create_dataloader(opt, world_size, rank)
#
# model = Pix2PixModel(opt)
# model.eval()
#
# synthesis_fdr = os.path.join(opt.results_dir, 'synthesis')
#
# if not os.path.exists(synthesis_fdr):
#     os.makedirs(synthesis_fdr)
# # test
# for i, data_i in enumerate(dataloader):
#     if i * opt.batchSize >= opt.how_many:
#         break
#     generated = model(data_i, mode='inference')
#
#     image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
#     synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))
#     synthesis_final_img.save(os.path.join(synthesis_fdr, data_i['path'][0].rsplit("/")[-1]))
#
# print("the end of resynthesis")

print("the beginning of mae process")
import shutil
import yaml
from torchvision.transforms import ToPILImage
# shutil.rmtree(os.path.join(args.show_dir, 'temp'))

import sys
import torchvision
sys.path.remove("/root/code_projects/tj-anormaly-seg/image_synthesis")
sys.path.insert(0,'/root/code_projects/tj-anormaly-seg')
print(os.getcwd())
print(sys.path)
from image_dissimilarity.util import trainer_util, metrics

############################################################################################
opt.results_dir = "/root/autodl-tmp/data_dis/preprocess/tj-sdro-train"
###########################################################################################

dataroot = opt.results_dir
soft_fdr = os.path.join(dataroot, 'mae_features')

if not os.path.exists(soft_fdr):
    os.makedirs(soft_fdr)

config_file_path = '/root/code_projects/tj-anormaly-seg/image_dissimilarity/configs/test/mytest_configuration.yaml'
# load experiment setting
with open(config_file_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# activate GPUs
gpu_ids = 0
config['gpu_ids'] = gpu_ids
gpu = int(gpu_ids)

# get data_loaders
cfg_train_loader = config['test_dataloader']
cfg_train_loader['dataset_args']['dataroot'] = dataroot
# get_dataloader function will plus the semantic and label value with 255 after resize and to tensor, and the aspect_ratio is to set the h and w ratio in resize function
train_loader = trainer_util.get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):  # why require_grad is false
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

from torch.nn.modules.upsampling import Upsample
up5 = Upsample(scale_factor=16, mode='bicubic')
up4 = Upsample(scale_factor=8, mode='bicubic')
up3 = Upsample(scale_factor=4, mode='bicubic')
up2 = Upsample(scale_factor=2, mode='bicubic')
up1 = Upsample(scale_factor=1, mode='bicubic')
to_pil = ToPILImage()

# Going through visualization loader
weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
vgg = VGG19().cuda(gpu)

with torch.no_grad():
    for i, data_i in enumerate(train_loader):
        print('Generating image %i out of %i' % (i + 1, len(train_loader)))
        img_name = os.path.basename(data_i['original_path'][0])
        original = data_i['original'].cuda(gpu)
        synthesis = data_i['synthesis'].cuda(gpu)

        x_vgg, y_vgg = vgg(original), vgg(synthesis)
        feat5 = torch.mean(torch.abs(x_vgg[4] - y_vgg[4]), dim=1).unsqueeze(1)
        feat4 = torch.mean(torch.abs(x_vgg[3] - y_vgg[3]), dim=1).unsqueeze(1)
        feat3 = torch.mean(torch.abs(x_vgg[2] - y_vgg[2]), dim=1).unsqueeze(1)
        feat2 = torch.mean(torch.abs(x_vgg[1] - y_vgg[1]), dim=1).unsqueeze(1)
        feat1 = torch.mean(torch.abs(x_vgg[0] - y_vgg[0]), dim=1).unsqueeze(1)

        img_5 = up5(feat5)
        img_4 = up4(feat4)
        img_3 = up3(feat3)
        img_2 = up2(feat2)
        img_1 = up1(feat1)

        combined = weights[0] * img_1 + weights[1] * img_2 + weights[2] * img_3 + weights[3] * img_4 + weights[
            4] * img_5
        min_v = torch.min(combined.squeeze())
        max_v = torch.max(combined.squeeze())
        combined = (combined.squeeze() - min_v) / (max_v - min_v)

        combined = to_pil(combined.cpu())
        pred_name = 'mea_' + img_name
        array = np.array(combined)
        combined.save(os.path.join(soft_fdr, pred_name))

