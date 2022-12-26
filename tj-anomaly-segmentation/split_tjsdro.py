#先在本机搜索得到标签名称,搜索只含有5、6、10号标签，而不含有1、2、3、4、7、8、9、11的图片

import os
import numpy as np
from PIL import Image
import shutil

# if os.path.exists("catalog_to_test.txt"):
#     os.remove("catalog_to_test.txt")
# fe = open("catalog_to_test.txt",'w')
# in_class = np.array([5,6,10])
# out_class = np.array([1,2,3,4,7,8,9,11])
# for root, dir, files in os.walk("G:\datasets\TJ_SDRO",topdown=False):
#     for name in files:
#         if 'label' in root:
#             if '.png' in name:
#                 if ('cutout' not in name) and ('pseudo' not in name) and ('foreground' not in name) and ('downloading' not in name):
#                     unique_class = np.unique(Image.open(os.path.join(root,name)))
#                     if np.any(np.isin(in_class,unique_class)) and (not np.any(np.isin(out_class,unique_class))):
#                         # if not np.isin(unique_class, 2)\
#                         #     and not np.isin(unique_class, 3)\
#                         #     and not np.isin(unique_class, 4)\
#                         #     and not np.isin(unique_class, 6)\
#                         #     and not np.isin(unique_class, 7)\
#                         #     and not np.isin(unique_class, 8)\
#                         #     and not np.isin(unique_class, 9)\
#                         #     and not np.isin(unique_class, 11):
#                         fe.write(name)
#                         fe.write('\r\n')
# fe.close()
# print("the collecting is finished")


# if os.path.exists("catalog_to_train.txt"):
#     os.remove("catalog_to_train.txt")
# fe = open("catalog_to_train.txt",'w')
# out_class = np.array([5,6,10])
# in_class = np.array([1,2,3,4,7,8,9,11])
# for root, dir, files in os.walk("G:\datasets\TJ_SDRO",topdown=False):
#     for name in files:
#         if 'label' in root:
#             if '.png' in name:
#                 if ('cutout' not in name) and ('pseudo' not in name) and ('foreground' not in name) and ('downloading' not in name):
#                     unique_class = np.unique(Image.open(os.path.join(root,name)))
#                     if np.any(np.isin(in_class,unique_class)) and (not np.any(np.isin(out_class,unique_class))):
#                         # if not np.isin(unique_class, 2)\
#                         #     and not np.isin(unique_class, 3)\
#                         #     and not np.isin(unique_class, 4)\
#                         #     and not np.isin(unique_class, 6)\
#                         #     and not np.isin(unique_class, 7)\
#                         #     and not np.isin(unique_class, 8)\
#                         #     and not np.isin(unique_class, 9)\
#                         #     and not np.isin(unique_class, 11):
#                         fe.write(name)
#                         fe.write('\r\n')
# fe.close()


# f = open("/root/code_projects/tj-anormaly-seg/catalog_to_train.txt")
# line = ""
# train_original_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro-train/original"
# train_labels_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro-train/labels"
# if not os.path.exists(train_original_path):
#     os.makedirs(train_original_path)
# if not os.path.exists(train_labels_path):
#     os.makedirs(train_labels_path)
#
# train_count = 0
# while True:
#     line = f.readline()
#     if not line:      #等价于if line == "":
#         break
#     line = line.replace("\n","")
#     if line == "":
#         continue
#     img_path = os.path.join("/root/autodl-tmp/data_dis/preprocess/tj-sdro/original", line).replace(".png","_leftImg8bit.png")
#     label_path = os.path.join("/root/autodl-tmp/data_dis/preprocess/tj-sdro/labels_with_ROI", line).replace(".png","_gtFine_labelIds.png")
#
#     if os.path.exists(img_path) and os.path.exists(label_path):
#         print(img_path)
#         train_count = train_count + 1
#         shutil.move(img_path, os.path.join(train_original_path, line.replace(".png","_leftImg8bit.png")))
#         shutil.move(label_path, os.path.join(train_labels_path, line.replace(".png","_gtFine_labelIds.png")))
# print("there are %d train images to move" % (train_count))
# f.close()


# f = open("/root/code_projects/tj-anormaly-seg/catalog_to_test.txt")
# line = ""
# test_original_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro-test/original"
# test_labels_path = "/root/autodl-tmp/data_dis/preprocess/tj-sdro-test/labels"
# if not os.path.exists(test_original_path):
#     os.makedirs(test_original_path)
# if not os.path.exists(test_labels_path):
#     os.makedirs(test_labels_path)
#
# test_count = 0
# while True:
#     line = f.readline()
#     if not line:      #等价于if line == "":
#         break
#     line = line.replace("\n","")
#     if line == "":
#         continue
#     img_path = os.path.join("/root/autodl-tmp/data_dis/preprocess/tj-sdro/original", line).replace(".png","_leftImg8bit.png")
#     label_path = os.path.join("/root/autodl-tmp/data_dis/preprocess/tj-sdro/labels_with_ROI", line).replace(".png","_gtFine_labelIds.png")
#
#     if os.path.exists(img_path) and os.path.exists(label_path):
#         print(img_path)
#         test_count = test_count + 1
#         shutil.move(img_path, os.path.join(test_original_path, line.replace(".png","_leftImg8bit.png")))
#         shutil.move(label_path, os.path.join(test_labels_path, line.replace(".png","_gtFine_labelIds.png")))
# print("there are %d test images to move" % (test_count))

# img = Image.open('/root/autodl-tmp/data_dis/preprocess/tj-sdro-test/labels/Image_20211021152209690_gtFine_labelIds.png')
# a = np.array(img)
# b = a.copy()
# b[np.where(a == 1)] = 255
# img = Image.fromarray(b.astype(np.uint8))
# img.save('/root/autodl-tmp/look_up.png')

total_counter = 0
counter = 0
for root, dir, files in os.walk("/root/autodl-tmp/data_dis/preprocess/tj-sdro-train/labels", topdown=False):
    for name in files:
        total_counter+=1
        img_array = np.array(Image.open(os.path.join(root, name)))
        if len(img_array.shape) == 3:
            counter += 1
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array[0]
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(root,name))
print(total_counter)
print(counter)