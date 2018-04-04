from PIL import Image
import os
import pandas as pd


Dir = "/home/yokoyang/PycharmProjects/untitled/new_data"
target_size = 896
folder_name = "/home/yokoyang/PycharmProjects/untitled/new_data/split-data/"

train_img = pd.read_csv(Dir + '/data_imageID.csv')

Image_ID = sorted(train_img.ImageId.unique())
name = 'target'
i = 0
for i in Image_ID:
    filename = os.path.join(Dir, name,'{}.tif'.format(i))
    img = Image.open(filename)
    img2 = img.crop((0, 0, target_size, target_size))
    pic_name = folder_name + i + ".tif"
    img2.save(pic_name)
