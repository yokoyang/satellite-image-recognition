import os

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image

# 一般建筑&农村&工厂&阴影
# 水体&植被
# 建筑场地&裸地
# 运动场&道路
# name_list = ['tree']
name_list = [ 'bare_land', 'building_yard', 'countryside', 'factory', 'general_building', 'playground',
             'road', 'shadow', 'tree', 'water']
# name = 'split-data'
# name = 'bare_land'
# name = 'building_yard'
# name = 'countryside'
# name = 'factory'
# name = 'general_building'
# name = 'playground'
# name = 'road'
# name = 'shadow'
# name = 'tree'
# name = 'water'
for name in name_list:
    # /home/yokoyang/PycharmProjects/untitled/biaozhu/water/0_0_0.tif
    # /home/yokoyang/PycharmProjects/untitled/biaozhu/water/0_0_0.tif
    Dir = "/home/yokoyang/PycharmProjects/untitled/biaozhu"
    target_size = 896
    folder_name = "/home/yokoyang/PycharmProjects/untitled/biaozhu/new/" + name + "/"

    train_img = pd.read_csv(Dir + '/data_imageID.csv')

    Image_ID = sorted(train_img.ImageId.unique())
    i = 0
    for i in Image_ID:
        filename = os.path.join(Dir, name, '{}.tif'.format(i))
        cv2_im = cv2.imread(filename)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im = cv2_im.astype(np.uint8)
        cv2_im ^= 255
        img = Image.fromarray(cv2_im)
        img2 = img.crop((0, 0, target_size, target_size))
        pic_name = folder_name + i + ".tif"
        tiff.imsave(pic_name, np.array(img2))


