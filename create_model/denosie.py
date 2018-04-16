import cv2 as cv2
import numpy as np


# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何
def getPixel(image, x, y, G, N):
    L = image.getpixel((x, y))
    if L > G:
        L = True
    else:
        L = False

    nearDots = 0

    if L == (image.getpixel((x - 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y + 1)) > G):
        nearDots += 1

    if nearDots < N:
        return image.getpixel((x, y - 1))
    else:
        return None

    # 降噪


# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点
# G: Integer 图像二值化阀值
# N: Integer 降噪率 0 <N <8
# Z: Integer 降噪次数
# 输出
#  0：降噪成功
#  1：降噪失败

def clearNoise(image, N, dis, size):
    height, width = image.shape[:2]
    counter = 0
    image_t = np.zeros([size, size], dtype=np.uint8)
    image_t[:, :] = image[:, :, 1]
    for x in range(dis, width - dis):
        for y in range(dis, height - dis):
            clolor_x_y = image_t[x, y]
            if image_t[x - 1, y - 1] != clolor_x_y:
                counter += 1
            if image_t[x - 1, y] != clolor_x_y:
                counter += 1
            if image_t[x - 1, y + 1] != clolor_x_y:
                counter += 1
            if image_t[x, y - 1] != clolor_x_y:
                counter += 1
            if image_t[x, y + 1] != clolor_x_y:
                counter += 1
            if image_t[x + 1, y - 1] != clolor_x_y:
                counter += 1
            if image_t[x + 1, y] != clolor_x_y:
                counter += 1

            if image_t[x + dis, y + dis] != clolor_x_y:
                counter += 1
            if image_t[x - dis, y - dis] != clolor_x_y:
                counter += 1
            if image_t[x + dis, y] != clolor_x_y:
                counter += 1
            if image_t[x + dis, y - dis] != clolor_x_y:
                counter += 1
            if image_t[x, y + dis] != clolor_x_y:
                counter += 1
            if image_t[x - dis, y] != clolor_x_y:
                counter += 1

            if counter > N:
                image_t ^= 1
                image[x, y, 0] = image_t[x, y]
                image[x, y, 1] = image_t[x, y]
                image[x, y, 2] = image_t[x, y]
    return image
    #
    # draw = ImageDraw.Draw(image)
    #
    # for i in range(0, Z):
    #     for x in range(1, image.size[0] - 1):
    #         for y in range(1, image.size[1] - 1):
    #             color = getPixel(image, x, y, G, N)
    #             if color != None:
    #                 draw.point((x, y), color)

    # 测试代码


def main():
    input_path = '../images/msk_prd_2.tif'
    output_path = '../images/msk_prd_remove_noise.tif'
    image = cv2.imread(input_path)

    #
    # # 打开图片
    # image = Image.open(input_path)
    # print(image)
    # 去噪,G = 50,N = 4,Z = 4

    cv2.imwrite(output_path, clearNoise(image, 4, 2, 1024))
    #
    # # 保存图片
    # image.save(output_path)
    # img = cv2.imread(input_path)
    # # img2 = img[:,np.newaxis]
    #
    # dst = cv2.fastNlMeansDenoising(img, h=0, templateWindowSize=7, searchWindowSize=21)
    # cv2.imwrite(output_path, dst)


if __name__ == '__main__':
    main()
