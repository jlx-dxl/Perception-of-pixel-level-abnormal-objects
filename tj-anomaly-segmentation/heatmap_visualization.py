import numpy as np
from PIL import Image
import cv2
import os

img_path = []
for root, dir, files in os.walk("G:\datasets\ev\Tj Sdro Test\pick_out",topdown=False):
    for name in files:
        img_path.append(os.path.join(root,name))

img_path.sort()
original_img = cv2.imread(img_path[0])
for i in range(1,6):
    heatmap = cv2.imread(img_path[i])
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img * 0.6
    cv2.imwrite(img_path[i].replace(".png","heatmap.png"), superimposed_img)