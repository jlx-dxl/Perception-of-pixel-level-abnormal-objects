import os
import PIL.Image as Image

original_save_path = "/root/autodl-tmp/data_dis/preprocess/fs_lost_and_found_val/original"
if not os.path.exists(original_save_path):
    os.makedirs(original_save_path)
for root, dir, files in os.walk("/root/autodl-tmp/data_dis/preprocess/fs_lost_and_found_val/labels_with_ROI",topdown=False):
    for name in files:
        image_path = name[5:].replace("_labels.png","_leftImg8bit.png")
        for root1,dir1,files1 in os.walk("/root/autodl-tmp/lost_and_found/leftImg8bit",topdown=False):
            for name1 in files1:
                if image_path == name1:
                    img = Image.open(os.path.join(root1, name1))
                    img.save(os.path.join(original_save_path, name[:5] + name1))