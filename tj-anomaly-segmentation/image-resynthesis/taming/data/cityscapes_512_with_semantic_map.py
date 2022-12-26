import os
import numpy as np
import cv2
import torch
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 semantic_instead=False,
                 interpolation="bicubic",
                 n_labels=34, shift_segmentation=False,
                 ):
        self.BoolTensor = torch.BoolTensor

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
                                   for l in self.image_paths],
            "instance_path_": [os.path.join(self.segmentation_root, l.replace("_leftImg8bit.png", "_gtFine_instanceIds.png"))
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

    def get_edges(self,t):
        edge = self.BoolTensor(
            t.size()).zero_()  # for PyTorch versions higher than 1.2.0, use BoolTensor instead of ByteTensor
        edge[:, :, 1:] = edge[:, :, 1:] | (t[:, :, 1:] != t[:, :, :-1])
        edge[:, :, :-1] = edge[:, :, :-1] | (t[:, :, 1:] != t[:, :, :-1])
        edge[:, 1:, :] = edge[:, 1:, :] | (t[:, 1:, :] != t[:, :-1, :])
        edge[:, :-1, :] = edge[:, :-1, :] | (t[:, 1:, :] != t[:, :-1, :])
        return edge.float()

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

        instance = Image.open(example["instance_path_"])
        instance = np.array(instance)
        instance = self.segmentation_rescaler(image=instance)["image"]
        instance = Image.fromarray(instance)
        if instance.mode == 'L':
            transform = transforms.ToTensor()
            instance_tensor = transform(instance) * 255
            instance_tensor = instance_tensor.long()

        else:
            transform = transforms.ToTensor()
            instance_tensor = transform(instance)

        instance = self.get_edges(instance_tensor)
        instance = torch.squeeze(instance)
        instance = np.array(instance).astype(np.uint8)

        onehot2 = np.eye(2)[instance]
        example["instance"] = onehot2
        return example

class Examples(SegmentationBase):
    def __init__(self, interpolation="bicubic",
                 semantic_instead=False,
                 data_csv="/media/group2/data/pengjianyi/taming-trans-cityscapes/train.txt",
                 segmentation_root="/media/group2/data/wanghaitao/cityscapes/leftImg8bit/train",
                 data_root="/media/group2/data/wanghaitao/cityscapes/gtFine/train"):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         semantic_instead=semantic_instead,
                         segmentation_root=segmentation_root,
                         interpolation=interpolation)