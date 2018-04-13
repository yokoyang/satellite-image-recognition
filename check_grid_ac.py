import numpy as np
import tifffile as tiff

dic_class = dict()
dic_class['water'] = [48, 93, 254]
dic_class['tree'] = [12, 169, 64]
dic_class['playground'] = [105, 17, 151]
dic_class['road'] = [111, 111, 111]
dic_class['building_yard'] = [255, 255, 255]
dic_class['bare_land'] = [239, 156, 119]
dic_class['general_building'] = [249, 255, 25]
dic_class['countryside'] = [227, 23, 33]
dic_class['factory'] = [48, 254, 254]
dic_class['shadow'] = [255, 1, 255]


def change_rgb_2_num(img):
    width, height, channel = img.shape
    result = np.zeros((width, height), dtype=np.uint)
    for w in range(width):
        for h in range(height):
            if np.array_equal(dic_class['water'], img[w][h]):
                result[w][h] = 1
            elif np.array_equal(dic_class['tree'], img[w][h]):
                result[w][h] = 2
            elif np.array_equal(dic_class['playground'], img[w][h]):
                result[w][h] = 3
            elif np.array_equal(dic_class['road'], img[w][h]):
                result[w][h] = 4
            elif np.array_equal(dic_class['building_yard'], img[w][h]):
                result[w][h] = 5
            elif np.array_equal(dic_class['bare_land'], img[w][h]):
                result[w][h] = 6
            elif np.array_equal(dic_class['general_building'], img[w][h]):
                result[w][h] = 7
            elif np.array_equal(dic_class['countryside'], img[w][h]):
                result[w][h] = 8
            elif np.array_equal(dic_class['factory'], img[w][h]):
                result[w][h] = 9
            elif np.array_equal(dic_class['shadow'], img[w][h]):
                result[w][h] = 10
    return result


def check_grid_acc(img_ground, img_pre, grid_size):
    width, height = img_pre.shape
    W = int(width // grid_size)
    H = int(height // grid_size)
    counter = 0
    all_cell = W * H
    for w_i in range(W):
        for h_j in range(H):
            y_img_ground = img_ground[grid_size * w_i:grid_size * (w_i + 1), grid_size * h_j:grid_size * (h_j + 1)]
            tu = sorted([(np.sum(y_img_ground == i), i) for i in set(y_img_ground.flat)])
            max_time_1, max_num_1 = (tu[-1])

            y_img_ground = img_pre[grid_size * w_i:grid_size * (w_i + 1), grid_size * h_j:grid_size * (h_j + 1)]
            tu = sorted([(np.sum(y_img_ground == i), i) for i in set(y_img_ground.flat)])
            max_time_1, max_num_2 = (tu[-1])
            if max_num_1 == max_num_2:
                counter += 1
    return counter / all_cell


img_id = '1_3.tif'
# img_id = '1_3.tif'
print(img_id)
img1 = change_rgb_2_num(tiff.imread('/home/yokoyang/PycharmProjects/untitled/check_result/' + img_id))
img2 = change_rgb_2_num(tiff.imread('/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-mask-data/' + img_id))
print(check_grid_acc(img1, img2, 16))
