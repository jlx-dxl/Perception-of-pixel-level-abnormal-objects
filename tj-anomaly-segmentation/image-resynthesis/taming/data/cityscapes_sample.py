import os
import numpy as np
import cv2
import torch
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class SegmentationBase(Dataset):
    def __init__(self,data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=34, shift_segmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_root = data_root
        self.segmentation_root = segmentation_root

        relative_file_path = []
        file_path = []
        segmentation_path = []
        for root, dirs, files in os.walk(data_root, topdown=False):
            for name in files:
                relative_file_path.append(os.path.join(root, name).rsplit("/")[-1])
                file_path.append(os.path.join(root, name))
                segmentation_path.append(os.path.join(root, "pred_mask_" + name).replace("_leftImg8bit.png", "_labelIds.png").replace("/leftImg8bit/", "/gtFine/"))

        # for root, dirs, files in os.walk(data_root, topdown=False):
        #     for name in files:
        #         relative_file_path.append(os.path.join(root, name).rsplit("/")[-1])
        #         file_path.append(os.path.join(root, name))
        #         segmentation_path.append(os.path.join(root, name).replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("/leftImg8bit/", "/gtFine/"))

        # for root, dirs, files in os.walk(data_root, topdown=False):
        #     for name in files:
        #         relative_file_path.append(os.path.join(root, name).rsplit("/")[-1])
        #         file_path.append(os.path.join(root, name))
        #         segmentation_path.append(os.path.join(root, name).replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("/1/", "/2/"))

        self.original_size = Image.open(file_path[0]).size

        self._length = len(file_path)
        self.labels = {
            "relative_file_path_": relative_file_path,
            "file_path_": file_path,
            "segmentation_path_": segmentation_path
        }

        # size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
        self.image_rescaler = albumentations.Resize(height=self.size, width=self.size,interpolation=self.interpolation)
        self.segmentation_rescaler = albumentations.Resize(height=self.size, width=self.size,interpolation=cv2.INTER_NEAREST)
            # self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, #将短边变成max_size，并保持长宽比,可以试试512
            #                                                      interpolation=self.interpolation)
            # self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
            #                                                             interpolation=cv2.INTER_NEAREST)
            # self.center_crop = not random_crop
            # if self.center_crop: #最好是random_crop
            #     self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            # else:
            #     self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            # self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        segmentation = Image.open(example["segmentation_path_"])
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]



        # if self.size is not None:
        #     processed = self.preprocessor(image=image,
        #                                   mask=segmentation
        #                                   )
        # else:
        processed = {"image": image,
                     "mask": segmentation
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = onehot
        return example

class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic",
                 segmentation_root="/media/group2/data/wanghaitao/cityscapes/leftImg8bit/train",
                 data_root="/media/group2/data/wanghaitao/cityscapes/gtFine/train"):
        super().__init__(data_root=data_root,
                         segmentation_root=segmentation_root,
                         size=size, random_crop=random_crop, interpolation=interpolation)