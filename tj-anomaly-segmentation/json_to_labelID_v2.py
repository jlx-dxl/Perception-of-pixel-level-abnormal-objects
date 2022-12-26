from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annFile='/media/fire/d/share_data/datasets/coco/annotations/person_keypoints_val2017.json'
coco=COCO(annFile) # display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
dataDir = '/media/fire/d/share_data/datasets/coco'
dataType = 'val2017'
I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
#plt.axis('off')
plt.imshow(I)
plt.show()

catIds = []
for ann in coco.dataset['annotations']:
    if ann['image_id'] == imgIds[0]:
        catIds.append(ann['category_id'])
plt.imshow(I);
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns)
plt.imshow(I);
plt.axis('off');
plt.show()


coco = COCO("/media/fire/d/share_data/datasets/coco/annotations/instances_val2017.json")

img_ids = coco.getImgIds()
print(len(img_ids))
cat_ids = []
for ann in coco.dataset["annotations"]:
    if ann["image_id"] == img_ids[0]:
        cat_ids.append(ann["category_id"])
ann_ids = coco.getAnnIds(imgIds=img_ids[0], catIds=cat_ids)
ann_ids2 = coco.getAnnIds(imgIds=img_ids[0], catIds=cat_ids)
plt.imshow(I)
print(ann_ids)
print(ann_ids2)
anns = coco.loadAnns(ann_ids)
coco.showAnns(anns)
plt.imshow(I)
plt.show()