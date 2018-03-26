import cv2
from PIL import Image

img = Image.open("suzhouSmall.tif")

target_size = 1024

width = img.size[0]
height = img.size[1]
col = width // target_size
row = height // target_size
for i in range(col):
    for j in range(row):
        img2 = img.crop((target_size * i, target_size * j, target_size * (i + 1), target_size * (j + 1)))
        pic_name = "split-data/" + str(i) + "_" + str(j) + ".tif"
        img2.save(pic_name)
cv2.imwrite()
