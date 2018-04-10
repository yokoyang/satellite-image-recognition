import cv2
import numpy as np
import tifffile as tiff
from u_net_img import U_net

n_split = 4
crop_size = 224
patch_size = 192
get_size = 159
test_file_dir = ''
msk_file_dir = ''
file_dir = ''

u_net = U_net(n_split, crop_size, patch_size, get_size, test_file_dir, msk_file_dir, file_dir)
all_class = ['countryside', 'playground', 'tree', 'road', 'building_yard', 'bare_land', 'water', 'general_building',
             'factory', 'shadow']

model = u_net.get_unet_multi()
model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/all.hdf5')

img = tiff.imread('/home/yokoyang/PycharmProjects/untitled/896_val/shanghai2/2_2.tif')
test_img = cv2.resize(img, (u_net.scale_size, u_net.scale_size))
img_RGB = test_img.astype(np.float32) / 255

prd_result = u_net.predict_10_class(model, img_RGB, test_img, n_class=10, th=0.1)

# plt.figure(figsize=[21, 8])
# plt.subplot(1, 2, 1)
# plt.title('Training Image')
# plt.imshow(test_img)
# plt.subplot(1, 2, 2)
# plt.title('predict Image')
prd_result = prd_result.astype(np.uint8)
tiff.imsave("result.tif", prd_result)
print(prd_result.shape)
# plt.imshow(prd_result)
# plt.show()
