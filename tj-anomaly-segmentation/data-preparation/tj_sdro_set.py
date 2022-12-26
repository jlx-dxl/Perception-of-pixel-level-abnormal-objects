import os
import PIL.Image as Image
import numpy as np
import shutil

original_save_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro/original"
if not os.path.exists(original_save_path):
    os.makedirs(original_save_path)

label_save_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro/labels_with_ROI"
if not os.path.exists(label_save_path):
    os.makedirs(label_save_path)
    # change the label
void_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for root, dir, files in os.walk("/root/autodl-tmp/TJ-SDRO",topdown=False):
    if "scene_28" in root:
        continue
    for name in files:
        if "label" in root:
            label_array = np.array(Image.open(os.path.join(root, name)))
            print(np.unique(label_array))
            label_img = np.zeros(label_array.shape)
            for void_label in void_labels:
                mask_unknown = np.where(label_array == void_label, 1, 0).astype(np.uint8)
                label_img += mask_unknown
            final_mask = np.where(label_img != 1, 0, 1).astype(np.uint8)
            mask_img = Image.fromarray((final_mask).astype(np.uint8))
            mask_img.save(os.path.join(label_save_path, name.replace(".png","_gtFine_labelIds.png")))
        else:
            shutil.copyfile(os.path.join(root,name), os.path.join(original_save_path,name.replace(".bmp","_leftImg8bit.png")))
