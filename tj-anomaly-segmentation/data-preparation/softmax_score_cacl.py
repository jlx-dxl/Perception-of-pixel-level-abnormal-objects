import torch
from torch.backends import cudnn
import yaml
from torchvision.transforms import ToPILImage, ToTensor
import torchvision
import os
import mmcv
import cv2
import argparse
import mmseg.datasets.custom
import glob
import math
from omegaconf import OmegaConf
import numpy as np
import albumentations
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint,BaseModule
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from PIL import Image
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config',
                        default='/root/code_projects/image-segmentation/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='/root/code_projects/tj-anormaly-seg/image-segmentation/pretrained/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show_dir',
        default='/root/autodl-tmp/data_dis/preprocess/1',
        type=str,
        help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

args = parse_args()

##################################################################################################################################
args.config = '/root/code_projects/tj-anormaly-seg/image-segmentation/segformer_mit-b5_8x1_1024x1024_160k_customdatasets.py'
args.show_dir = '/root/autodl-tmp/data_dis/preprocess/tj-sdro-test'
####################################################################################################################################


args.work_dir = '/root/autodl-tmp/segmentation/test'
args.checkpoint = '/root/code_projects/tj-anormaly-seg/image-segmentation/pretrained/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth' #load model from here
# args.aug-test = False
args.out = None
# args.format-only
args.eval = 'None'
args.show = False
# args.gpu-collect
args.tmpdir = '/root/autodl-tmp/seg-tmp'
args.options = None
args.launcher = 'none'
#args.local_rank
#below are things not occur in the parse_args()


# assert args.out or args.eval or args.format_only or args.show \
#        or args.show_dir, \
#     ('Please specify at least one operation (save/eval/format/show the '
#      'results / save the results) with the argument "--out", "--eval"'
#      ', "--format-only", "--show" or "--show_dir"')

if 'None' in args.eval:
    args.eval = None

if args.eval and args.format_only:

    raise ValueError('--eval and --format_only cannot be both specified')

if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    raise ValueError('The output file must be a pkl file.')

cfg = mmcv.Config.fromfile(args.config)

if args.options is not None:
    cfg.merge_from_dict(args.options)

#if set true，there might be a randomness when forward
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

if args.aug_test:
    if cfg.data.test.type == 'CityscapesDataset':
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.
        ]
        cfg.data.test.pipeline[1].flip = True
    elif cfg.data.test.type == 'ADE20KDataset':
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.75, 0.875, 1.0, 1.125, 1.25
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True

cfg.model.pretrained = None
cfg.data.test.test_mode = True
cfg.data.test['data_root'] = args.show_dir


########################################################################################
cfg.data.test['pipeline'][1]['img_scale'] = (1440, 1080)
cfg.data.test['img_suffix'] = '.png'
#########################################################################################


# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

# build the dataloader
# TODO: support multiple images per gpu (only minor changes are needed)



dataset = build_dataset(cfg.data.test) #the val set of cityscapes
# Create save directory
if not os.path.exists(args.show_dir):
    os.makedirs(args.show_dir)

softmax_prediction_fdr = os.path.join(args.show_dir, 'tja_softmax_prediction')
if not os.path.exists(softmax_prediction_fdr):
    os.makedirs(softmax_prediction_fdr)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)


#######################################################################################
h = 1440
w = 1080
flat_pred = np.zeros(w * h * len(data_loader), dtype='float32')
flat_labels = np.zeros(w * h * len(data_loader), dtype='float32')
#######################################################################################



# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')) # the test_cfg is whole
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES'] #继承checkpoint的class，即类别名称
model.PALETTE = checkpoint['meta']['PALETTE'] #继承checkpoint的palette，即对应类别的颜色

# # to check whether it has something to do with the label transform
# overlap_original_fdr = os.path.join(args.show_dir, 'overlap_original')
# if not os.path.exists(overlap_original_fdr):
#     os.makedirs(overlap_original_fdr)
# original_palette=[[0,  0,  0],[0,  0,  0],[0,  0,  0],[0,  0,  0],[0,  0,  0],[111, 74,  0],[81,  0, 81],
#                [128, 64, 128], [244, 35, 232],[250,170,160],[230,150,140],[70, 70, 70], [102, 102, 156],
#                [190, 153, 153], [180,165,180], [150,100,100],[150,120, 90], [153, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
#                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
#                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],[0,  0, 90],
#                [0,  0,110], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
# original_palettes = []
# for i in original_palette:
#     for j in i:
#         original_palettes.append(j)
# zero_pad = 256 * 3 - len(original_palettes)
# for i in range(zero_pad):
#     original_palettes.append(0)

palette = []
for i in dataset.PALETTE:
    for j in i:
        palette.append(j)
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

efficient_test = False
if args.eval_options is not None:
    efficient_test = args.eval_options.get('efficient_test', False)

if not distributed:
    model = MMDataParallel(model, device_ids=[1])
    # outputs = single_gpu_test(model, data_loader, False, None,
    #                           efficient_test)

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
        pred = np.array(result[0][0]).squeeze()

        # get entropy
        softmax = torch.tensor(np.array(result[0][1]))
        softmax_pred = torch.sum(-softmax*torch.log(softmax), dim=1)
        softmax_pred = (softmax_pred - softmax_pred.min()) / softmax_pred.max()
        softmax_pred_og = softmax_pred.detach().cpu().numpy().squeeze()

        ########################################################################################################################
        label_img = Image.open(
            data['img_metas'][0].data[0][0]['filename'].replace("/original/", "/labels_with_ROI/").replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"))
        ########################################################################################################################
        label = np.array(label_img)

        if len(label.shape) == 3:
            label = label.transpose(2, 0, 1)
            label = label[0]

        soft_result = Image.fromarray(softmax_pred_og)
        soft_result.save(os.path.join(softmax_prediction_fdr,data['img_metas'][0].data[0][0]['filename']))

        softmax_flatten = softmax_pred_og.flatten().squeeze()
        label_flatten = label.flatten().squeeze()
        flat_pred[batch_indices[0] * w * h:batch_indices[0] * w * h + w * h] = softmax_flatten
        flat_labels[batch_indices[0] * w * h:batch_indices[0] * w * h + w * h] = label_flatten



else:
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                             args.gpu_collect, efficient_test)


print('Calculating metric scores')
invalid_indices = np.argwhere(flat_labels == 255)
flat_labels = np.delete(flat_labels, invalid_indices)
flat_pred = np.delete(flat_pred, invalid_indices)

sys.path.insert(0,'/root/code_projects/tj-anormaly-seg/image_dissimilarity')
from util import metrics
results = metrics.get_metrics(flat_labels, flat_pred)

print("roc_auc_score : " + str(results['auroc']))
print("mAP: " + str(results['AP']))
print("FPR@95%TPR : " + str(results['FPR@95%TPR']))


print('Segmentation Results saved.')



