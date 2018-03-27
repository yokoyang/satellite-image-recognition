import os

from PIL import Image

img_name = "/home/yokoyang/Downloads/kaggle-data/shanghai2.tif"
folder_name = "/home/yokoyang/PycharmProjects/untitled/896_val/shanghai2/"
img = Image.open(img_name)
target_size = 896

width = img.size[0]
height = img.size[1]
col = width // target_size
row = height // target_size

for i in range(col):
    for j in range(row):
        img2 = img.crop((target_size * i, target_size * j, target_size * (i + 1), target_size * (j + 1)))
        pic_name = folder_name + str(i) + "_" + str(j) + ".tif"
        img2.save(pic_name)
