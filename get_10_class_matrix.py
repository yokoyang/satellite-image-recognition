import os

import numpy as np
import pandas as pd
import tifffile as tiff

dic_class = dict()
dic_class['water'] = [48, 93, 254]
dic_class['tree'] = [12, 169, 64]
dic_class['playground'] = [102, 17, 151]
dic_class['road'] = [111, 111, 111]
dic_class['building_yard'] = [255, 255, 255]
dic_class['bare_land'] = [239, 156, 119]
dic_class['general_building'] = [249, 255, 25]
dic_class['countryside'] = [227, 22, 33]
dic_class['factory'] = [48, 254, 254]
dic_class['shadow'] = [255, 1, 255]

tag_name = 'split-mask-data'
class_name = 'mix_all'

n_class = 10


def get_all_mask(img):
    width, height = img.shape[:2]
    msk = np.zeros((width, height, n_class), dtype=np.uint8)

    for w in range(width):
        for h in range(height):
            i = 0
            for class_key, value in dic_class.items():
                if np.array_equal(value, img[w][h]):
                    msk[w][h][i] = 1
                    continue
                i += 1

    return msk


# Dir = "/home/yokoyang/PycharmProjects/untitled/896_val"
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
    filename = os.path.join(Dir, tag_name, '{}.tif'.format(img_id))
    img = tiff.imread(filename)
    msk_file_name = os.path.join(Dir, class_name, '{}.npy'.format(img_id))
    msk_img = get_all_mask(img)
    np.save(msk_file_name, msk_img)
    print(msk_img[0][0][3])
