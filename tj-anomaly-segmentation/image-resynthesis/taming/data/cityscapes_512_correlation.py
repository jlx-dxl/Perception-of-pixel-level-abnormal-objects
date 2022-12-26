import os
import numpy as np
import cv2
import torch
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 interpolation="bicubic",
                 n_labels=34, shift_segmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
                                   for l in self.image_paths]
        }

        self.interpolation = interpolation
        self.interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
        self.image_rescaler = albumentations.Resize(height=256, width=512,interpolation=self.interpolation)
        self.segmentation_rescaler = albumentations.Resize(height=256, width=512,interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.image_rescaler(image=image)["image"]
        segmentation = Image.open(example["segmentation_path_"])
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        segmentation = self.segmentation_rescaler(image=segmentation)["image"]

        processed = {"image": image,
                     "mask": segmentation
                     }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = onehot
        return example

class Examples(SegmentationBase):
    def __init__(self, interpolation="bicubic",
                 data_csv="/media/group2/data/pengjianyi/taming-trans-cityscapes/train.txt",
                 segmentation_root="/media/group2/data/wanghaitao/cityscapes/leftImg8bit/train",
                 data_root="/media/group2/data/wanghaitao/cityscapes/gtFine/train"):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         interpolation=interpolation)