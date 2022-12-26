import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image,ImageDraw
import os
import json
scene2path="G:/datasets/tj-sdro/scene_2"
scene6path="G:/datasets/tj-sdro/scene_6"
scene7path="G:/datasets/tj-sdro/scene_7"



def getJson(filepath):
    '''从文件夹获取json文件内容，返回字典'''
    jsonfile=filepath + "label/annotations.json"
    jsonstr = open(jsonfile, "r", encoding="utf8").read()
    d_json = json.loads(jsonstr)
    # print(d_json)
    return d_json


def getPath():
    '''输入图片文件夹路径'''
    filepath = scene7path
    if filepath.endswith != "/" or filepath.endswith != "\\":
        filepath = filepath + "/"
    return filepath


filepath = getPath()
d_json = getJson(filepath)
# print(d_json)
image_json = d_json['images']
image_map={}
for key in image_json:
    image_map[key['id']] = key['file_name']
atat_json = d_json['annotations']
image = []
curr_id = 0

for key in atat_json:
    # print(key)
    # print(data)
    if key['image_id'] != curr_id:
        curr_id = key['image_id']
        if(len(image)) < curr_id:
            image.append({})
        image[curr_id - 1]['pictureName'] = image_map[key['image_id']]
        for i in range(1,13):
            image[curr_id - 1][i] = []
    image[curr_id - 1][key['category_id']].insert(-1,key['segmentation'][0])

# width = d_json['images'][0]['height']
# height = d_json['images'][0]['width']
# dpi = 100  # 分辨率
# ycwidth = width / dpi  # 宽度(英寸) = 像素宽度 / 分辨率
# ycheight = height / dpi  # 高度(英寸) = 像素高度 / 分辨率

color_code=[]
color_code.append('#000000')
color_code.append('#010101')
color_code.append('#020202')
color_code.append('#030303')
color_code.append('#040404')
color_code.append('#050505')
color_code.append('#060606')
color_code.append('#070707')
color_code.append('#080808')
color_code.append('#090909')
color_code.append('#0A0A0A')
color_code.append('#0B0B0B')
color_code.append('#0C0C0C')

for key in image:
    # fig, ax = plt.subplots(figsize=(ycwidth, ycheight))
    img = Image.new('RGB', [1440, 1080], color_code[0])
    img1 = ImageDraw.Draw(img)
    for i in range(1,13):
        if key[12 - i + 1] == []:
            continue
        for region in key[12 - i + 1]:
            # region.append(region[0])
            # region.append(region[1])
            #region = np.array(region)
            #xy = region.reshape(-1, 2)
            img1.polygon(region, fill=color_code[12 - i + 1], outline=color_code[12 - i + 1])
    path = filepath + "label/" + key['pictureName']
    path1 = filepath + key['pictureName']
    path2 = path.rsplit(".", 1)[0] + ".png"
    if os.path.exists(path1):
        img.save(path2)
    #         plt.plot(1440 - xy[:, 0], 1080 - xy[:, 1])
    #         plt.fill_between(1440 - xy[:, 0], 1080 - xy[:, 1])  # 对该分割区域填充颜色
    #
    # plt.show()
    # plt.xticks([0, width])
    # plt.yticks([0, -height])
    # # plt.axis([0,0,1,1])
    # plt.axis("off")
    # # 保存图片
    #
    # # print(sourcePicture)
    # # print(path)
    # # plt.savefig(path + "-mask.png", format='png', bbox_inches='tight', transparent=False, dpi=100) # bbox_inches='tight' 图片边界空白紧致, 背景透明
    # # plt.savefig(path + "-mask.png", format='png', transparent=True, dpi=100)
    # # plt.show()
    #
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.show()
    # path = filepath + key['pictureName']
    # path2 = path.rsplit(".", 1)[0] + "-mask.png"
    # fig.savefig(path2, format='png', transparent=False, dpi=100, pad_inches=0)

