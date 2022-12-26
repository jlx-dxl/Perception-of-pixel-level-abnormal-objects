import os
import numpy as np
from PIL import Image

from natsort import natsorted

import sys


def convert_gtCoarse_to_labels(data_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    semantic_paths = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            temp_path = os.path.join(root,name)
            if '_labelIds' in temp_path:
                semantic_paths.append(temp_path)


    # semantic_paths = [os.path.join(data_path, image)
    #                   for image in os.listdir(data_path) if 'labelTrainIds' in image]

    semantic_paths = natsorted(semantic_paths) # 以一种自然的方式排序

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))

        # get mask where instance is located
        print(np.unique(semantic_img))
        a = np.zeros(semantic_img.shape,int)
        b = np.ones(semantic_img.shape,int)
        mask = np.where(((semantic_img == 6) | (semantic_img == 5) | (semantic_img == 4) | (semantic_img == 3) | (semantic_img == 2) | (semantic_img == 1) | (semantic_img == 0)), b, a)
        mask_img = Image.fromarray((mask).astype(np.uint8))
        semantic_name = os.path.basename(semantic)
        mask_img.save(os.path.join(save_dir, 'labels', semantic_name))


def convert_gtCoarse_to_labels_ROI(data_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels_with_ROI')):
        os.mkdir(os.path.join(save_dir, 'labels_with_ROI'))

    semantic_paths = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            temp_path = os.path.join(root,name)
            if '_labelIds' in temp_path:
                semantic_paths.append(temp_path)


    # semantic_paths = [os.path.join(data_path, image)
    #                   for image in os.listdir(data_path) if 'labelTrainIds' in image]

    semantic_paths = natsorted(semantic_paths) # 以一种自然的方式排序

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))

        # get mask where instance is located
        print(np.unique(semantic_img))
        a = np.zeros(semantic_img.shape,int)
        b = np.ones(semantic_img.shape,int)
        c = b * 255

        # below is the function test
        # a = np.zeros((4, 8),int)
        # b = np.ones((4, 8),int)
        # c = b * 255
        # semantic_img = np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31]])
        # mask1 = np.where(((semantic_img == 6) | (semantic_img == 5) | (semantic_img == 4) | (semantic_img == 3) | (semantic_img == 2) | (semantic_img == 1) | (semantic_img == 0)), b, a)
        # mask2 = np.where(((semantic_img == 9) | (semantic_img == 10) | (semantic_img == 14) | (semantic_img == 15) | (
        #             semantic_img == 16) | (semantic_img == 18) | (semantic_img == 29) | (semantic_img == 30)), c, a)
        # mask = mask2 + mask1

        mask1 = np.where(((semantic_img == 6) | (semantic_img == 5) | (semantic_img == 4) | (semantic_img == 3) | (semantic_img == 2) | (semantic_img == 1) | (semantic_img == 0)), b, a)
        mask2 = np.where(((semantic_img == 9) | (semantic_img == 10) | (semantic_img == 14) | (semantic_img == 15) | (
                    semantic_img == 16) | (semantic_img == 18) | (semantic_img == 29) | (semantic_img == 30)), c, a)
        # mask2 = np.where(semantic_img == 9, 10, 14, 15, 16, 18, 29, 30)
        # mask3 = mask2.copy()
        # mask3[np.where(mask2 == 1)] = 255
        mask = mask2 + mask1
        mask_img = Image.fromarray((mask).astype(np.uint8))
        semantic_name = os.path.basename(semantic)
        mask_img.save(os.path.join(save_dir, 'labels_with_ROI', semantic_name))

if __name__ == '__main__':
    data_path = '/root/autodl-tmp/cityscapes_segformer/gtFine/val'
    save_dir = '/root/autodl-tmp/data_dis/preprocess/1'
    #save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process'
    convert_gtCoarse_to_labels_ROI(data_path, save_dir)

    #semantic_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/semantic_labelids'
    #convert_semantic_to_trainids(semantic_path, save_dir)