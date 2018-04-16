from PIL import Image
import os

img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/row.tif"
# img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/msk.tif"
# img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/shanghai_label.tif"
# img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/shanghai_label.tif"

# img_name = "/home/yokoyang/Downloads/kaggle-data/row.tif"
# folder_name = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-mask-data-fix/"
# folder_name = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu_new/split-mask-data/"
folder_name = "/home/yokoyang/PycharmProjects/untitled/640_biaozhu/split-data/"
# folder_name = "/home/yokoyang/PycharmProjects/untitled/640_biaozhu/split-mask-data/"

# folder_name_val = "/home/yokoyang/PycharmProjects/untitled/896_val/split-mask-data-road/"
# folder_name = "/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-data/"
# folder_name_val = "/home/yokoyang/PycharmProjects/untitled/896_val/split-data/"
# Dir = "/home/yokoyang/PycharmProjects/untitled/images"
# Dir = "/home/yokoyang/PycharmProjects/untitled/biaozhu/水体&植被"

img = Image.open(img_name)
target_size = 640

width = img.size[0]
height = img.size[1]
col = width // target_size
row = height // target_size

img_tag = ''
for i in range(col):
    for j in range(row):
        img2 = img.crop((target_size * i, target_size * j, target_size * (i + 1), target_size * (j + 1)))
        pic_name = folder_name + img_tag + str(i) + "_" + str(j) + ".tif"
        img2.save(pic_name)
img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/row2.tif"
# img_name = "/home/yokoyang/PycharmProjects/untitled/shanghai_model/msk2.tif"
img = Image.open(img_name)
width = img.size[0]
height = img.size[1]
col = width // target_size
row = height // target_size
img_tag = 'msk2_'
for i in range(col):
    for j in range(row):
        img2 = img.crop((target_size * i, target_size * j, target_size * (i + 1), target_size * (j + 1)))
        pic_name = folder_name + img_tag + str(i) + "_" + str(j) + ".tif"
        img2.save(pic_name)

# for i in range(10):
#     image_id = "0_0_" + str(i)
#     filename = os.path.join(Dir, '{}.tif'.format(image_id))
#     img = Image.open(filename)
#     img2 = img.crop((0, 0, target_size, target_size))
#     pic_name = folder_name + image_id + ".tif"
#     img2.save(pic_name)
