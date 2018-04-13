import os

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import morphology

# water 0,108,255
# tree 0,168,62
# playground 102,34,153
# road 112,112,112
# building_yard 255, 255, 255
# bare_land 242, 155, 118
# general_building 249,255,25
# countryside 227,22,33
# factory 48,254,254
# shadow 255,0,255

dic_class = dict()
# [48, 93, 254]
# dic_class['water'] = [0, 108, 255]
# # tree [12, 169, 64]
# dic_class['tree'] = [0, 168, 62]
# dic_class['playground'] = [102, 34, 153]
# dic_class['road'] = [112, 112, 112]
# dic_class['building_yard'] = [255, 255, 255]
# dic_class['bare_land'] = [242, 155, 118]
# dic_class['general_building'] = [249, 255, 25]
#
# dic_class['countryside'] = [227, 22, 33]
# dic_class['factory'] = [48, 254, 254]
# dic_class['shadow'] = [255, 0, 255]
dic_class['water'] = [48, 93, 254]
dic_class['tree'] = [12, 169, 64]
dic_class['playground'] = [105, 17, 151]
dic_class['road'] = [111, 111, 111]
dic_class['building_yard'] = [255, 255, 255]
dic_class['bare_land'] = [239, 156, 119]
dic_class['general_building'] = [249, 255, 25]
dic_class['countryside'] = [227, 23, 33]
dic_class['factory'] = [48, 254, 254]
dic_class['shadow'] = [255, 1, 255]
# 一般建筑&农村&工厂&阴影
# 运动场&道路
# 水体&植被
# 建筑场地&裸地

tag_name = 'split-mask-data'
# tag_name = '水体&植被'          water
# class_name = 'water'
# class_name = 'tree'
# class_name = 'playground'
# class_name = 'road'
# class_name = 'building_yard'
# class_name = 'bare_land'
# class_name = 'general_building'
# class_name = 'countryside'
# class_name = 'factory'
class_name = 'shadow'

print(class_name)


def get_mask(img, img_class):
    width, height = img.shape[:2]
    channel = 3
    msk = np.zeros((width, height, channel))
    for w in range(width):
        for h in range(height):
            if np.array_equal(dic_class[img_class], img[w][h]):
                msk[w][h] = [255, 255, 255]
    return msk


# Dir = "/home/yokoyang/PycharmProjects/untitled/new_data"
Dir = "/home/yokoyang/PycharmProjects/untitled/new_data"


def get_files_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L.append(os.path.splitext(file)[0])
    return L


def image2csv(img_folder_name, dir_name, csv_name):
    L = get_files_name(img_folder_name)
    df = pd.DataFrame()
    df['ImageId'] = L
    df.to_csv(dir_name + "/" + csv_name, index=False, header=True)


img_folder_name = Dir + '/' + tag_name
image2csv(img_folder_name, Dir, "data_imageID.csv")
train_img = pd.read_csv(Dir + '/data_imageID.csv')

Image_ID = sorted(train_img.ImageId.unique())

for i, img_id in enumerate(Image_ID):
    # if img_id[0:3] != '0_0':
    #     continue
    print(i)
    filename = os.path.join(Dir, tag_name, '{}.tif'.format(img_id))
    img = tiff.imread(filename)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msk_file_name = os.path.join(Dir, class_name, '{}.tif'.format(img_id))
    msk_img = get_mask(img, class_name)
    # cv2.imwrite(msk_file_name, msk_img)
    msk_img = msk_img > 1
    ms = msk_img[:, :, 1]
    dst = morphology.remove_small_objects(ms, min_size=2, connectivity=1)
    # dst = dst.astype(np.float32)
    dst = dst.astype(np.uint8)
    dst ^= 1
    msk_img = msk_img.astype(np.uint8)
    msk_img[:, :, 0] = dst[:, :] * 255
    msk_img[:, :, 1] = dst[:, :] * 255
    msk_img[:, :, 2] = dst[:, :] * 255
    tiff.imsave(msk_file_name, msk_img)
