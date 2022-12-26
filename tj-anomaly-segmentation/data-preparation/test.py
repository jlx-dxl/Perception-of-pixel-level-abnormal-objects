import os
from PIL import Image
import numpy as np
# relative_file_path = []
file_path = []
# segmentation_path = []
classes = []
for root, dirs, files in os.walk("/root/autodl-tmp/data_dis/preprocess/1/temp/gtFine/val", topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img)
        for sample in np.unique(img_array):
            if classes.count(sample) == 0 :
                classes.append(sample)
            if sample == 7:
                print("road exist")
            elif sample == 8:
                print("sidewalk exist")
            elif sample == 11:
                print("building exist")
            elif sample == 12:
                print("wall exist")
            elif sample == 13:
                print("fence exist")
            elif sample == 17:
                print("pole exist")
classes.sort()
print(classes)


        # segmentation_path.append(os.path.join(root, name).replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
    # for name in dirs:
    #     print(os.path.join(root, name))