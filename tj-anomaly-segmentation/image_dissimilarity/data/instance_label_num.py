import os
import numpy as np
from PIL import Image


for root, dir, files in os.walk('/root/autodl-tmp/cityscapes_segformer/gtFine/train',topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if '_instanceIds' in path:
            img = Image.open(path)
            img_array = np.array(Img)
            print(np.unique(img))
            # for i in range(img_array.shape[0]):
            #     for j in range(img_array.shape[1]):
            #         # if(img_array[i][j][0] != img_array[i][j][1] or img_array[i][j][1] != img_array[i][j][2] or img_array[i][j][0] != img_array[i][j][2]):
            #         print("found a cord is %d %d %d" %(img_array[i][j][0], img_array[i][j][1], img_array[i][j][2]))
            # unique_class = np.unique(img_array)
            # print(unique_class.shape)
            # unique_class.sort()
            # print(unique_class)
#             for i in np.unique(img_array):
#                 if unique_class.count(i) == 0:
#                     print("class %d is append in" % i)
#                     unique_class.append(i)
#
# print(unique_class.sort())