import torch
from torch.backends import cudnn
import yaml
from torchvision.transforms import ToPILImage, ToTensor
import torchvision
import os
import mmcv
import cv2
import argparse
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
args.config = '/root/code_projects/tj-anormaly-seg/image-segmentation/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
args.work_dir = '/root/autodl-tmp/segmentation/test'
args.checkpoint = '/root/code_projects/tj-anormaly-seg/image-segmentation/pretrained/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth' #load model from here
# args.aug-test = False
args.out = None
# args.format-only
args.eval = 'None'
args.show = False
print(args.show_dir)
args.show_dir = '/root/autodl-tmp/data_dis/preprocess/1'
# args.gpu-collect
args.tmpdir = '/root/autodl-tmp/seg-tmp'
args.options = None
args.launcher = 'none'
#args.local_rank
#below are things not occur in the parse_args()



# # assert args.out or args.eval or args.format_only or args.show \
# #        or args.show_dir, \
# #     ('Please specify at least one operation (save/eval/format/show the '
# #      'results / save the results) with the argument "--out", "--eval"'
# #      ', "--format-only", "--show" or "--show_dir"')
#
# if 'None' in args.eval:
#     args.eval = None
#
# if args.eval and args.format_only:
#
#     raise ValueError('--eval and --format_only cannot be both specified')
#
# if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
#     raise ValueError('The output file must be a pkl file.')
#
# cfg = mmcv.Config.fromfile(args.config)
#
# if args.options is not None:
#     cfg.merge_from_dict(args.options)
#
# #if set true，there might be a randomness when forward
# if cfg.get('cudnn_benchmark', False):
#     torch.backends.cudnn.benchmark = True
#
# if args.aug_test:
#     if cfg.data.test.type == 'CityscapesDataset':
#         # hard code index
#         cfg.data.test.pipeline[1].img_ratios = [
#             0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
#         ]
#         cfg.data.test.pipeline[1].flip = True
#     elif cfg.data.test.type == 'ADE20KDataset':
#         # hard code index
#         cfg.data.test.pipeline[1].img_ratios = [
#             0.75, 0.875, 1.0, 1.125, 1.25
#         ]
#         cfg.data.test.pipeline[1].flip = True
#     else:
#         # hard code index
#         cfg.data.test.pipeline[1].img_ratios = [
#             0.5, 0.75, 1.0, 1.25, 1.5, 1.75
#         ]
#         cfg.data.test.pipeline[1].flip = True
#
# cfg.model.pretrained = None
# cfg.data.test.test_mode = True
#
# # init distributed env first, since logger depends on the dist info.
# if args.launcher == 'none':
#     distributed = False
# else:
#     distributed = True
#     init_dist(args.launcher, **cfg.dist_params)
#
# # build the dataloader
# # TODO: support multiple images per gpu (only minor changes are needed)
# dataset = build_dataset(cfg.data.test) #the val set of cityscapes
# # Create save directory
# if not os.path.exists(args.show_dir):
#     os.makedirs(args.show_dir)
#
# color_mask_fdr = os.path.join(args.show_dir, 'color-mask')
# overlap_fdr = os.path.join(args.show_dir, 'overlap')
# semantic_label_fdr = os.path.join(args.show_dir, 'semantic_labelIds')
# semantic_fdr = os.path.join(args.show_dir, 'semantic')
# original_fdr = os.path.join(args.show_dir, 'original')
# soft_fdr = os.path.join(args.show_dir, 'entropy')
# soft_fdr_2 = os.path.join(args.show_dir, 'logit_distance')
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
# if not os.path.exists(original_fdr):
#     os.makedirs(original_fdr)
#
# if not os.path.exists(soft_fdr):
#     os.makedirs(soft_fdr)
#
# if not os.path.exists(soft_fdr_2):
#     os.makedirs(soft_fdr_2)
#
# # creates temporary folder to adapt format to image synthesis which use cityscapes datasets
# if not os.path.exists(os.path.join(args.show_dir, 'temp')):
#     os.makedirs(os.path.join(args.show_dir, 'temp'))
#     os.makedirs(os.path.join(args.show_dir, 'temp', 'gtFine', 'val'))
#     os.makedirs(os.path.join(args.show_dir, 'temp', 'leftImg8bit', 'val'))
#
# data_loader = build_dataloader(
#     dataset,
#     samples_per_gpu=1,
#     workers_per_gpu=cfg.data.workers_per_gpu,
#     dist=distributed,
#     shuffle=False)
#
# # build the model and load checkpoint
# cfg.model.train_cfg = None
# model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')) # the test_cfg is whole
# checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
# model.CLASSES = checkpoint['meta']['CLASSES'] #继承checkpoint的class，即类别名称
# model.PALETTE = checkpoint['meta']['PALETTE'] #继承checkpoint的palette，即对应类别的颜色
#
# palette = []
# for i in dataset.PALETTE:
#     for j in i:
#         palette.append(j)
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)
#
# efficient_test = False
# if args.eval_options is not None:
#     efficient_test = args.eval_options.get('efficient_test', False)
#
# if not distributed:
#     model = MMDataParallel(model, device_ids=[0])
#     # outputs = single_gpu_test(model, data_loader, False, None,
#     #                           efficient_test)
#
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     # The pipeline about how the data_loader retrieval samples from dataset:
#     # sampler -> batch_sampler -> indices
#     # The indices are passed to dataset_fetcher to get data from dataset.
#     # data_fetcher -> collate_fn(dataset[index]) -> data_sample
#     # we use batch_sampler to get correct data idx
#     loader_indices = data_loader.batch_sampler
#
#     for batch_indices, data in zip(loader_indices, data_loader):
#         # with torch.no_grad():
#         #     result = model(return_loss=False, **data)
#         # batch_size = len(result)
#         # for _ in range(batch_size):
#         #     prog_bar.update()
#         # pred = np.array(result[0][0]).squeeze()
#         #
#         # # get entropy
#         # softmax = torch.tensor(np.array(result[0][1]))
#         # softmax_pred = torch.sum(-softmax*torch.log(softmax), dim=1)
#         # softmax_pred = (softmax_pred - softmax_pred.min()) / softmax_pred.max()
#         #
#         # # get logit distance
#         # distance, _ = torch.topk(softmax, 2, dim=1)
#         # max_logit = distance[:, 0, :, :]
#         # max2nd_logit = distance[:, 1, :, :]
#         # result = max_logit - max2nd_logit
#         # map_logit = 1 - (result - result.min()) / result.max()
#         #
#         # softmax_pred_og = softmax_pred.cpu().numpy().squeeze()
#         # map_logit = map_logit.cpu().numpy().squeeze()
#         # softmax_pred_og = softmax_pred_og * 255
#         # map_logit = map_logit * 255
#         # pred_name = 'entropy_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # pred_name_2 = 'distance_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # # cv2.imwrite(os.path.join(soft_fdr, pred_name), softmax_pred_og)
#         # # cv2.imwrite(os.path.join(soft_fdr_2, pred_name_2), map_logit)
#         #
#         #
#         # label_out = pred.copy() #长点儿心吧
#         # full_path = os.path.realpath(__file__)
#         # sys.path.insert(0, full_path)
#         # from cityscapes_labels import label2trainid
#         # for label_id, train_id in label2trainid.items():
#         #     label_out[np.where(pred == train_id)] = label_id
#         # sys.path.remove(full_path)
#         #
#         # color_name = 'color_mask_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # overlap_name = 'overlap_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # pred_name = 'pred_mask_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # # colorized = Image.fromarray(pred.astype(np.uint8)).convert('P')
#         # # colorized.putpalette(palette)
#         img = Image.open(data['img_metas'][0].data[0][0]['filename']).convert('RGB')
#         img.save(os.path.join(original_fdr, data['img_metas'][0].data[0][0]['filename'].rsplit("/")[-1]))
#         path_tmp = os.path.join(args.show_dir, 'temp', 'leftImg8bit', 'val', data['img_metas'][0].data[0][0]['filename'].rsplit("/")[-1].rsplit(".")[0] + '_leftImg8bit.png')
#         img.save(path_tmp)
#         print("finish")
#         # # overlap = cv2.addWeighted(np.array(img), 0.5, np.array(colorized.convert('RGB')), 0.5, 0)
#         # # cv2.imwrite(os.path.join(overlap_fdr, overlap_name), overlap[:, :, ::-1])
#         # # cv2.imwrite(os.path.join(semantic_fdr, pred_name), pred)
#         # cv2.imwrite(os.path.join(args.show_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'), label_out)
#         # cv2.imwrite(os.path.join(args.show_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'), label_out)
#
#         # pred_name = 'pred_mask_' + data['img_metas'][0].data[0][0]['filename'].rsplit('/')[-1]
#         # label_out = Image.open(os.path.join(args.show_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'))
#         # label_out.save(os.path.join(args.show_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'))
#
# else:
#     model = MMDistributedDataParallel(
#         model.cuda(),
#         device_ids=[torch.cuda.current_device()],
#         broadcast_buffers=False)
#     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
#                              args.gpu_collect, efficient_test)

#the segmentation process is over, now comes the resynthesis
print('Segmentation Results saved.')



print('Starting Image Synthesis Process')

full_path = os.path.realpath(__file__)
image_resyn_path = os.path.join(os.path.dirname(full_path),'../image-resynthesis')
sys.path.insert(0, image_resyn_path)

# from main import instantiate_from_config, DataModuleFromConfig
# from torch.utils.data.dataloader import default_collate
# import torch.nn.functional as F
#
# def load_model_from_config(config, sd, gpu=True, eval_mode=True):
#     if "ckpt_path" in config.params:
#         print("Deleting the restore-ckpt path from the config...")
#         config.params.ckpt_path = None
#     if "downsample_cond_size" in config.params:
#         print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
#         config.params.downsample_cond_size = -1
#         config.params["downsample_cond_factor"] = 0.5
#     try:
#         if "ckpt_path" in config.params.first_stage_config.params:
#             config.params.first_stage_config.params.ckpt_path = None
#             print("Deleting the first-stage restore-ckpt path from the config...")
#
#         # to use none overfitting cond stage model disable this
#         # if "ckpt_path" in config.params.cond_stage_config.params:
#         #     config.params.cond_stage_config.params.ckpt_path = None
#         #     print("Deleting the cond-stage restore-ckpt path from the config...")
#     except:
#         pass
#
#     model = instantiate_from_config(config)
#     if sd is not None:
#         missing, unexpected = model.load_state_dict(sd, strict=False)
#         if "ckpt_path" in config.params.cond_stage_config.params:
#             cond = torch.load(config.params.cond_stage_config.params.ckpt_path, map_location='cpu')
#             cond_sd = cond["state_dict"]
#             model.cond_stage_model.load_state_dict(cond_sd, strict=False)
#         print(f"Missing Keys in State Dict: {missing}")
#         print(f"Unexpected Keys in State Dict: {unexpected}")
#     if gpu:
#         model.cuda()
#     if eval_mode:
#         model.eval()
#     return {"model": model}
#
#
# def get_data(config):
#     # get data
#     data = instantiate_from_config(config.data)
#     data.prepare_data()
#     data.setup()
#     return data
# # @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def load_model_and_dset(config, ckpt, gpu, eval_mode):
#     # get data
#     dsets = get_data(config)   # calls data.config ...
#
#     # now load the specified checkpoint
#     if ckpt:
#         pl_sd = torch.load(ckpt, map_location="cpu")
#         global_step = pl_sd["global_step"]
#     else:
#         pl_sd = {"state_dict": None}
#         global_step = None
#     model = load_model_from_config(config.model,
#                                    pl_sd["state_dict"],
#                                    gpu=gpu,
#                                    eval_mode=eval_mode)["model"]
#     return dsets, model, global_step
#
# def get_parser_resyn():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-r",
#         "--resume",
#         type=str,
#         nargs="?",
#         help="load from logdir or checkpoint in logdir",
#     )
#     parser.add_argument(
#         "-b",
#         "--base",
#         nargs="*",
#         metavar="base_config.yaml",
#         help="paths to base configs. Loaded from left-to-right. "
#         "Parameters can be overwritten or added with command-line options of the form `--key value`.",
#         default=list(),
#     )
#     parser.add_argument(
#         "-c",
#         "--config",
#         nargs="?",
#         metavar="single_config.yaml",
#         help="path to single config. If specified, base configs will be ignored "
#         "(except for the last one if left unspecified).",
#         const=True,
#         default="",
#     )
#     parser.add_argument(
#         "--ignore_base_data",
#         action="store_true",
#         help="Ignore data specification from base configs. Useful if you want "
#         "to specify a custom datasets on the command line.",
#     )
#     return parser
#
# parser = get_parser_resyn()
#
# opt, unknown = parser.parse_known_args()
#
# opt.resume = '/root/autodl-tmp/synthesis/logs/2022-05-05T12-47-10_cityscapes_scene_images_transformer'
# opt.base = '/root/code_projects/tj-anormaly-seg/image-resynthesis/configs/cityscapes_transformer_sample.yaml'
#
# ckpt = None
# if opt.resume:
#     if not os.path.exists(opt.resume):
#         raise ValueError("Cannot find {}".format(opt.resume))
#     if os.path.isfile(opt.resume):
#         paths = opt.resume.split("/")
#         try:
#             idx = len(paths) - paths[::-1].index("logs") + 1
#         except ValueError:
#             idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
#         logdir = "/".join(paths[:idx])
#         ckpt = opt.resume
#     else:
#         assert os.path.isdir(opt.resume), opt.resume
#         logdir = opt.resume.rstrip("/")
#         ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
#     print(f"logdir:{logdir}")
#     base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
#     opt.base = [opt.base]
#     for config in base_configs:
#         opt.base.insert(0,config)
#
#
# if opt.config:
#     if type(opt.config) == str:
#         opt.base = [opt.config]
#     else:
#         opt.base = [opt.base[-1]]
#
# configs = [OmegaConf.load(cfg) for cfg in opt.base]
#
# cli = OmegaConf.from_dotlist(unknown)
# if opt.ignore_base_data:
#     for config in configs:
#         if hasattr(config, "data"): del config["data"]
# config = configs[-1]
#
# config.data.params.test.params.data_root = os.path.join(args.show_dir,'temp', 'leftImg8bit', 'val')
# config.data.params.test.params.segmentation_root = os.path.join(args.show_dir, 'temp', 'gtFine', 'val')
#
# gpu = True
# eval_mode = True
#
# dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
#
# dset = dsets.datasets['test']
# batch_size = 20 # dsets.batch_size
#
# start_index = 0
# image_original_size = dset.original_size
# synthesis_fdr = os.path.join(args.show_dir, 'synthesis')
# if not os.path.exists(synthesis_fdr):
#     os.makedirs(synthesis_fdr)
#
#
# temp_color_fdr = os.path.join(args.show_dir, 'temp','color')
# if not os.path.exists(temp_color_fdr):
#     os.makedirs(temp_color_fdr)
#
# mask_resize = albumentations.Resize(height=image_original_size[1], width=image_original_size[0],
#                                    interpolation=cv2.INTER_NEAREST)
#
#
# img_resize = albumentations.Resize(height=image_original_size[1], width=image_original_size[0],
#                                    interpolation=cv2.INTER_CUBIC)
#
# # full_path = os.path.realpath(__file__)
# # sys.path.insert(0, full_path)
# # from cityscapes_labels import label2trainid
# # for root, dirs, files in os.walk(config.data.params.test.params.segmentation_root, topdown=False):
# #     for name in files:
# #         img_array = np.array(Image.open(os.path.join(root,name)).convert('RGB'))
# #         pred = img_array[0]
# #         label_out = pred
# #         for label_id, train_id in label2trainid.items():
# #             label_out[np.where(pred == train_id)] = label_id
# #         cv2.imwrite(os.path.join(root,name), label_out)
# # check_path = sys.path.pop(0)
# # print(check_path)
#
#
# while start_index < len(dset):
#     if start_index + batch_size > len(dset):
#         indices = list(range(start_index, len(dset) - 1))
#     else:
#         indices = list(range(start_index, start_index + batch_size))
#
#     example = default_collate([dset[i] for i in indices])
#     c = model.get_input("segmentation", example).to(model.device) #TODO: 想办法改变输出到正常分辨率，看看效果
#                                                                   # 同时检查label的对应情况，
#                                                                   # 还有就是GPU的利用率也得看看，然后加上torch.nograd()
#     with torch.no_grad() :
#         quant_c, c_indices = model.encode_to_c(c)
#         z_start_indices = c_indices[:, :0]
#         index_sample = model.sample(z_start_indices, c_indices,
#                                    steps=c_indices.shape[1],
#                                    sample=False)
#         x_sample_det = model.decode_to_img(index_sample, quant_c.shape)
#         cond_rec = model.cond_stage_model.decode(quant_c)
#         if model.cond_stage_key == "segmentation":
#             # get image from segmentation mask
#             num_classes = cond_rec.shape[1]
#
#             c = torch.argmax(c, dim=1, keepdim=True)
#             c = F.one_hot(c, num_classes=num_classes)
#             c = c.squeeze(1).permute(0, 3, 1, 2).float()
#             c = model.cond_stage_model.to_rgb(c)
#
#             cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
#             cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
#             cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
#             cond_rec = model.cond_stage_model.to_rgb(cond_rec)
#
#     x_sample_det = torch.clamp(x_sample_det, -1., 1.).to('cpu')
#     x_sample_det = (x_sample_det + 1.0) / 2.0
#     x_sample_det = x_sample_det.transpose(1,2).transpose(2,3)
#     x_sample_det = x_sample_det.numpy()
#     x_sample_det = (x_sample_det * 255)
#
#     c = torch.clamp(c, -1., 1.).to('cpu')
#     c = (c + 1.0) / 2.0
#     c = c.transpose(1,2).transpose(2,3)
#     c = c.numpy()
#     c = (c * 255)
#
#     cond_rec = torch.clamp(cond_rec, -1., 1.).to('cpu')
#     cond_rec = (cond_rec + 1.0) / 2.0
#     cond_rec = cond_rec.transpose(1,2).transpose(2,3)
#     cond_rec = cond_rec.numpy()
#     cond_rec = (cond_rec * 255)
#
#     for indice in indices:
#         x_output = img_resize(image=x_sample_det[indice - start_index])['image']
#         image_tmp = Image.fromarray(np.uint8(x_output))
#         image_tmp.save(os.path.join(synthesis_fdr, dset.labels['relative_file_path_'][indice]))
#
#         c_output = mask_resize(image=c[indice - start_index])['image']
#         # c_output = c[indice - start_index]
#         c_tmp = Image.fromarray(np.uint8(c_output))
#         c_tmp.save(os.path.join(temp_color_fdr,"origc_" + dset.labels['relative_file_path_'][indice]))
#
#         cond_rec_output = mask_resize(image=cond_rec[indice - start_index])['image']
#         # cond_rec_output = cond_rec[indice - start_index]
#         cond_rec_tmp = Image.fromarray(np.uint8(cond_rec_output))
#         cond_rec_tmp.save(os.path.join(temp_color_fdr, "recc_" + dset.labels['relative_file_path_'][indice]))
#
#     if start_index + batch_size >= len(dset):
#         start_index = len(dset)
#     else:
#         start_index += batch_size



print("the end of resynthesis")

print("the beginning of mae process")
import shutil
# shutil.rmtree(os.path.join(args.show_dir, 'temp'))
sys.path.remove(image_resyn_path)
sys.path.insert(0, os.path.join(full_path, "../image_dissimilarity"))
from util import trainer_util, metrics

dataroot = args.show_dir
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




