import os
import numpy as np
import PIL.Image as Image
counter = 0
# for root, dir, files in os.walk("/root/autodl-tmp/data_dis/preprocess/tj-sdro/original",topdown=False):
#     for name in files:
#         path = os.path.join(root, name)
#         label_path = path.replace("original","labels_with_ROI").replace("_leftImg8bit","_gtFine_labelIds")
#         if os.path.exists(label_path):
#             continue
#         else:
#             os.remove(path)
#             counter += 1
# print(counter)

for root, dir, files in os.walk("/root/autodl-tmp/data_dis/preprocess/tj-sdro/labels_with_ROI",topdown=False):
    for name in files:
        path = os.path.join(root, name)
        label_path = path.replace("labels_with_ROI","original").replace("_gtFine_labelIds","_leftImg8bit")
        if os.path.exists(label_path):
            continue
        else:
            os.remove(path)
            counter += 1
print(counter)

for root, dir, files in os.walk("/root/autodl-tmp/data_dis/preprocess/fs_lost_and_found_val/labels_with_ROI",topdown=False):
    for name in files:
        path = os.path.join(root, name)
        classes = np.unique(Image.open(path))
        print(classes)