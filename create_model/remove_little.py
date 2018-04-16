import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt


# 编写一个函数来生成原始二值图像
def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  # 生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  # 随机数种子
    points = l * generator.rand(2, n ** 2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l / (4. * n))  # 高斯滤波
    return mask > mask.mean()


data = microstructure(l=128)  # 生成测试图片

dst = morphology.remove_small_objects(data, min_size=300, connectivity=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(data, plt.cm.gray, interpolation='nearest')
ax2.imshow(dst, plt.cm.gray, interpolation='nearest')

fig.tight_layout()
plt.show()
