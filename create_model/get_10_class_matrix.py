import os

import numpy as np
import pandas as pd
import tifffile as tiff

dic_class = dict()
list_class = list()

dic_class['water'] = [48, 93, 254]
list_class.append(dic_class['water'])
dic_class['tree'] = [12, 169, 64]
list_class.append(dic_class['tree'])

dic_class['playground'] = [105, 17, 151]
list_class.append(dic_class['playground'])

dic_class['road'] = [111, 111, 111]
list_class.append(dic_class['road'])

dic_class['building_yard'] = [255, 255, 255]
list_class.append(dic_class['building_yard'])

dic_class['bare_land'] = [239, 156, 119]
list_class.append(dic_class['bare_land'])

dic_class['general_building'] = [249, 255, 25]
list_class.append(dic_class['general_building'])

dic_class['countryside'] = [227, 23, 33]
list_class.append(dic_class['countryside'])

dic_class['factory'] = [48, 254, 254]
list_class.append(dic_class['factory'])

dic_class['shadow'] = [255, 1, 255]
list_class.append(dic_class['shadow'])

tag_name = 'split-mask-data'
class_name = 'mix_all'

n_class = 10


def get_all_mask(img):
    width, height = img.shape[:2]
    msk = np.zeros((width, height), dtype=np.uint8)

    for w in range(width):
        for h in range(height):
            # i = 0
            for index, rgb_color in enumerate(list_class):
                if np.array_equal(rgb_color, img[w][h]):
                    msk[w][h] = index
            # for class_key, value in dic_class.items():
            #     if np.array_equal(value, img[w][h]):
            #         msk[w][h] = i
            #         break
            #     i += 1

    return msk


Dir = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu"


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
    print(i)
    filename = os.path.join(Dir, tag_name, '{}.tif'.format(img_id))
    img = tiff.imread(filename)
    msk_file_name = os.path.join(Dir, class_name, '{}.npy'.format(img_id))
    msk_img = get_all_mask(img)
    np.save(msk_file_name, msk_img)
