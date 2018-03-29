import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from keras import backend as K
from keras.backend import binary_crossentropy
from keras.layers import concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Nadam

N_split = 4

Patch_size = 192
crop_size = 224
edge_size = int((crop_size - Patch_size) / 2)
Scale_Size = Patch_size * N_split
n_class = 10

dic_class = dict()
dic_class['water'] = [48, 93, 254]
dic_class['tree'] = [12, 169, 64]
dic_class['playground'] = [102, 17, 151]
dic_class['road'] = [111, 111, 111]
dic_class['building_yard'] = [255, 255, 255]
dic_class['bare_land'] = [239, 156, 119]
dic_class['general_building'] = [249, 255, 25]
dic_class['countryside'] = [227, 22, 33]
dic_class['factory'] = [48, 254, 254]
dic_class['shadow'] = [255, 1, 255]


def get_image_row(image_id, dir_name):
    filename = os.path.join(
        Dir, dir_name, '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    return img


def get_image_without_msk(image_id, dir_name):
    img = get_image_row(image_id, dir_name)
    img = img.astype(np.float32) / 255
    img_RGB = cv2.resize(img, (Scale_Size, Scale_Size))
    return img_RGB


def reflect_img(img):
    reflect = cv2.copyMakeBorder(img, int(edge_size), int(edge_size), int(edge_size), int(edge_size),
                                 cv2.BORDER_REFLECT)
    return reflect


smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_unet1():
    inputs = Input((crop_size, crop_size, 3))
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_uniform')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_uniform')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_uniform')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_uniform')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((edge_size, edge_size), (edge_size, edge_size)))(conv9)
    conv9 = BatchNormalization(axis=1)(crop9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    return model


def predict_id(img, model, th, dir_name):
    prd = np.zeros((Patch_size * N_split, Patch_size * N_split, 1)).astype(np.float32)

    for i in range(N_split):
        for j in range(N_split):
            x = img[Patch_size * i:Patch_size * (i + 1), Patch_size * j:Patch_size * (j + 1), :]
            x = reflect_img(x)
            tmp = model.predict(x[None, :, :, :], batch_size=4)
            prd[Patch_size * i:Patch_size * (i + 1), Patch_size * j:Patch_size * (j + 1)] = tmp
    prd_result = prd > th
    return prd_result


def save_result_pic(filename, img):
    img = img.astype(np.uint8)
    img *= 255
    cv2.imwrite(filename, img)


def check_predict_without_mask(model, th, dir_name, img, Class_Type=1):
    msk_prd = predict_id(img, model, th, dir_name)
    print("without")
    img = get_image_without_msk(img_id, dir_name)
    msk_prd = msk_prd[:, :, 0]
    plt.figure(figsize=[21, 8])

    plt.subplot(1, 2, 1)
    plt.title('Training Image')
    plt.imshow(img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("img_result/train" + img_id + ".tif", img2)

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(msk_prd, cmap=plt.get_cmap('gist_ncar'))
    plt.axis('off')
    plt.show()


# In predicting testing dataset, need to use the same mean and std in preprocessing training data
def post_normalize_image(img, mean=0.338318, std=0.189734):
    img = (img - mean) / std
    return img


Dir = '/home/yokoyang/PycharmProjects/untitled/896_val'
train_img = pd.read_csv(Dir + '/data_imageID.csv')
# load all weights
building_model = get_unet1()
building_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/building.hdf5')

dir_name = "shanghai2"
img_id = '2_1'
img = get_image_without_msk(img_id, dir_name)
img = post_normalize_image(img)
# check_predict_without_mask(building_model, 0.5, dir_name, img)

build_msk = predict_id(img=img, model=building_model, th=0.5, dir_name=dir_name)
build_msk = build_msk[:, :, 0]

width, height = build_msk.shape
channel = 3

tree_model = get_unet1()
tree_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/tree.hdf5')
tree_msk = predict_id(img=img, model=tree_model, th=0.5, dir_name=dir_name)
tree_msk = tree_msk[:, :, 0]


water_model = get_unet1()
water_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/water.hdf5')
water_msk = predict_id(img=img, model=water_model, th=0.4, dir_name=dir_name)
water_msk = water_msk[:, :, 0]

row_img = get_image_row(img_id, dir_name)
row_img = cv2.resize(row_img, (Scale_Size, Scale_Size))

merge_msk = np.copy(row_img)
for i in range(width):
    for j in range(height):
        if build_msk[i][j]:
            merge_msk[i][j] = dic_class['general_building']
        elif water_msk[i][j]:
            merge_msk[i][j] = dic_class['water']
        elif tree_msk[i][j]:
            merge_msk[i][j] = dic_class['tree']

print("finished")
msk_file_dir = '/home/yokoyang/PycharmProjects/untitled/predict_img_result'
msk_file_name = msk_file_dir + "/" + img_id + ".tif"
tiff.imsave(msk_file_name, merge_msk)
