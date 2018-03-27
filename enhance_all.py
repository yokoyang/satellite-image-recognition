import os

import pandas as pd
from PIL import Image
from PIL import ImageEnhance

Dir = '/home/yokoyang/Downloads/kaggle-data/EU2'

train_img = pd.read_csv(Dir + '/2.csv')

Image_ID = sorted(train_img.ImageId.unique())


def contrast_image(image_id):
    filename = os.path.join(
        Dir, 'satellite', '{}.tif'.format(image_id))
    image = Image.open(filename)
    # 对比度增强
    contrast = 1.3
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted = enh_con.enhance(contrast)

    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    image_sharped = 3.0
    image_sharped = enh_sha.enhance(image_sharped)

    save_name = os.path.join(
        Dir, 'enhanced', '{}.tif'.format(image_id))
    image_sharped.save(save_name)
    return


for img_id in Image_ID:
    contrast_image(img_id)
