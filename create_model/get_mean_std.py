import os
import random

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

N_split = 4

Patch_size = 224
crop_size = 288
edge_size = int((crop_size - Patch_size) / 2)
Dir = '/home/yokoyang/PycharmProjects/untitled/new_data'

train_img = pd.read_csv(Dir + '/data_imageID.csv')

Image_ID = sorted(train_img.ImageId.unique())
Scale_Size = Patch_size * N_split


def reflect_img(img):
    reflect = cv2.copyMakeBorder(img, int(edge_size), int(edge_size), int(edge_size), int(edge_size),
                                 cv2.BORDER_REFLECT)
    return reflect


def get_image(image_id):
    filename = os.path.join(
        Dir, 'split-data', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = img.astype(np.float32) / 255
    img_RGB = cv2.resize(img, (Scale_Size, Scale_Size))
    return img_RGB


def get_mask(image_id):
    filename = os.path.join(
        Dir, 'general_building', '{}.tif'.format(image_id))
    msk = tiff.imread(filename)
    msk = msk.astype(np.float32) / 255
    msk = cv2.resize(msk, (Scale_Size, Scale_Size))
    msk_img = np.zeros([Scale_Size, Scale_Size], dtype=np.uint8)
    msk_img[:, :] = msk[:, :, 1]
    msk_img ^= 1
    return msk_img


def rotate_img(img, ang, size):
    rows = size
    cols = size
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 * ang, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def rotate_msk(msk, ang):
    return np.rot90(msk, ang)


def get_patch(img_id, pos=1):
    img_ = []
    msk_ = []
    img = get_image(img_id)
    img = reflect_img(img)
    mask = get_mask(img_id)
    for i in range(N_split):
        for j in range(N_split):
            y = mask[Patch_size * i:Patch_size * (i + 1), Patch_size * j:Patch_size * (j + 1)]
            if ((pos == 1) and (np.sum(y) > 0)) or (pos == 0):
                x_start = int(Patch_size * i)
                x_end = int(Patch_size * (i + 1) + edge_size * 2)
                y_start = int(Patch_size * j)
                y_end = int(Patch_size * (j + 1) + edge_size * 2)
                x = img[x_start:x_end, y_start:y_end, :]
                # start rotate y and x
                rdm = random.uniform(-2, 5)
                if rdm > 1:
                    ang = rdm // 1
                    x = rotate_img(x, ang, crop_size)
                    y = rotate_msk(y, ang)
                    # print(x.shape)
                    # print(y.shape)

                img_.append(x)
                msk_.append(y[:, :, None])

    return img_, msk_


def get_all_patches(pos=1):
    img_all = []
    msk_all = []
    count = 0
    for img_id in Image_ID:
        img_, msk_ = get_patch(img_id, pos=pos)
        if len(msk_) > 0:
            count = count + 1
            if count == 1:
                img_all = img_
                msk_all = msk_
            else:
                img_all = np.concatenate((img_all, img_), axis=0)
                msk_all = np.concatenate((msk_all, msk_), axis=0)

    # if pos == 1:
    #     np.save(Dir + '/output/data_pos_%d_%d_class%d' % (crop_size, N_split, Class_Type), img_all)
    #
    # else:
    #     np.save(Dir + '/output/data_%d_%d_class%d' % (crop_size, N_split, Class_Type), img_all)

    return img_all, msk_all[:, :, :, 0]


def get_normalized_patches():
    img_all, msk_all = get_all_patches()
    #     data = np.load(Dir + '/output/data_pos_%d_%d_class%d.npy' % (Patch_size, N_split, Class_Type))
    img = img_all
    msk = msk_all
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    print(mean, std)
    # print(np.mean(img), np.std(img))
    # return img, msk


get_normalized_patches()
