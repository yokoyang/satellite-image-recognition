import os

from PIL import Image

# img_name = "/home/yokoyang/Downloads/kaggle-data/msk.tif"
img_name = "/home/yokoyang/Downloads/kaggle-data/row.tif"
# folder_name = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-mask-data-road/"
# folder_name_val = "/home/yokoyang/PycharmProjects/untitled/896_val/split-mask-data-road/"
folder_name = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-data/"
folder_name_val = "/home/yokoyang/PycharmProjects/untitled/896_val/split-data/"
Dir = "/home/yokoyang/PycharmProjects/untitled/images"
# Dir = "/home/yokoyang/PycharmProjects/untitled/biaozhu/运动场&道路"
img = Image.open(img_name)
target_size = 896

width = img.size[0]
height = img.size[1]
col = width // target_size
row = height // target_size

for i in range(col):
    for j in range(row):
        img2 = img.crop((target_size * i, target_size * j, target_size * (i + 1), target_size * (j + 1)))

        if (i == 1 and j < 3) or (j == 7 and i < 6):
            pic_name = folder_name_val + str(i) + "_" + str(j) + ".tif"
        else:
            pic_name = folder_name + str(i) + "_" + str(j) + ".tif"
        img2.save(pic_name)

i = 0
for i in range(10):
    image_id = "0_0_" + str(i)
    filename = os.path.join(Dir, '{}.tif'.format(image_id))
    img = Image.open(filename)
    img2 = img.crop((0, 0, target_size, target_size))
    pic_name = folder_name + image_id + ".tif"
    img2.save(pic_name)
