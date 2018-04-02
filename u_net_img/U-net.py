import gc
import os
import random

import cv2
import keras
import numpy as np
import pandas as pd
import tifffile as tiff
from keras import backend as K
from keras.backend import binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.layers import concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split


class U_net(object):
    def __int__(self):
        self.smooth = 1e-12
        self.n_split = 4
        self.crop_size = 224
        self.patch_size = 192
        self.edge_size = int((self.crop_size - self.patch_size) / 2)
        self.get_size = 144
        self.scale_size = self.patch_size * self.n_split

    @property
    def scale_size(self):
        return self.scale_size

    @scale_size.setter
    def scale_size(self, value):
        if not isinstance(value, int):
            raise ValueError("scale size must be an integer!")
        if value < 0:
            raise ValueError("scale size must be positive!")
        self.scale_size = value

    @property
    def n_split(self):
        return self.n_split

    @n_split.setter
    def n_split(self, value):
        if not isinstance(value, int):
            raise ValueError("split number must be an integer!")
        if value < 0:
            raise ValueError("split number must be positive!")
        self.n_split = value

    @property
    def patch_size(self):
        return self.patch_size

    @patch_size.setter
    def patch_size(self, value):
        if not isinstance(value, int):
            raise ValueError("patch size must be an integer!")
        if value < 0:
            raise ValueError("patch size must be positive!")
        self.scale_size = value

    @property
    def crop_size(self):
        return self.crop_size

    @crop_size.setter
    def crop_size(self, value):
        self.crop_size = value

    @property
    def mean(self):
        return self.mean

    @mean.setter
    def mean(self, value):
        self.mean = value

    @property
    def std(self):
        return self.mean

    @std.setter
    def std(self, value):
        self.mean = value

    @property
    def edge_size(self):
        return self.edge_size

    @edge_size.setter
    def edge_size(self, value):
        self.edge_size = value

    @property
    def file_dir(self):
        return self.file_dir

    @file_dir.setter
    def file_dir(self, value):
        self.file_dir = value

    def get_mask(self, image_id, name_class):
        filename = os.path.join(
            self.file_dir, name_class, '{}.tif'.format(image_id))
        msk = tiff.imread(filename)
        msk = msk.astype(np.float32) / 255
        msk = cv2.resize(msk, (self.scale_size, self.scale_size))
        msk_img = np.zeros([self.scale_size, self.scale_size], dtype=np.uint8)
        msk_img[:, :] = msk[:, :, 1]
        msk_img ^= 1
        return msk_img

    def get_image(self, image_id):
        filename = os.path.join(
            self.file_dir, 'split-data', '{}.tif'.format(image_id))
        img = tiff.imread(filename)
        img = img.astype(np.float32) / 255
        img_RGB = cv2.resize(img, (self.scale_size, self.scale_size))
        return img_RGB

    def reflect_img(self, img):
        reflect = cv2.copyMakeBorder(img, self.edge_size, self.edge_size, self.edge_size, self.edge_size,
                                     cv2.BORDER_REFLECT)
        return reflect

    def rotate_img(self, img, ang, size):
        rows = size
        cols = size
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 * ang, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def rotate_msk(self, msk, ang):
        return np.rot90(msk, ang)

    def get_patch(self, img_id, name_class, pos=1):
        img_ = []
        msk_ = []
        img = self.get_image(img_id)
        img = self.reflect_img(img)
        mask = self.get_mask(img_id, name_class)
        for i in range(self.n_split):
            for j in range(self.n_split):
                y = mask[self.patch_size * i:self.patch_size * (i + 1), self.patch_size * j:self.patch_size * (j + 1)]
                if ((pos == 1) and (np.sum(y) > 0)) or (pos == 0):
                    x_start = int(self.patch_size * i)
                    x_end = int(self.patch_size * (i + 1) + self.edge_size * 2)
                    y_start = int(self.patch_size * j)
                    y_end = int(self.patch_size * (j + 1) + self.edge_size * 2)
                    x = img[x_start:x_end, y_start:y_end, :]
                    # start rotate y and x
                    rdm = random.uniform(-2, 5)
                    if rdm > 1:
                        ang = rdm // 1
                        x = self.rotate_img(x, ang, self.crop_size)
                        y = self.rotate_msk(y, ang)
                        # print(x.shape)
                        # print(y.shape)

                    img_.append(x)
                    msk_.append(y[:, :, None])

        return img_, msk_

    def get_all_patches(self, name_class, Image_ID, pos=1):
        img_all = []
        msk_all = []
        count = 0
        for img_id in Image_ID:
            img_, msk_ = self.get_patch(img_id, name_class, pos=pos)
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

    def get_normalized_patches(self, name_class, Image_ID):
        img_all, msk_all = self.get_all_patches(name_class, Image_ID)
        #     data = np.load(Dir + '/output/data_pos_%d_%d_class%d.npy' % (Patch_size, N_split, Class_Type))
        img = img_all
        msk = msk_all
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        print(mean, std)
        # print(np.mean(img), np.std(img))
        return img, msk

    def jaccard_coef(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        return K.mean(jac)

    def jaccard_coef_int(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))

        intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        return K.mean(jac)

    def jaccard_coef_loss(self, y_true, y_pred):
        return -K.log(self.jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

    def get_unet0(self):
        #     Patch_size = 224
        inputs = Input((self.crop_size, self.crop_size, 3))

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)

        up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)

        up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)

        up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        cropping_2d = Cropping2D(cropping=((self.edge_size, self.edge_size), (self.edge_size, self.edge_size)),
                                 input_shape=(int(self.crop_size), int(self.crop_size), 3))(conv10)

        model = Model(inputs=inputs, outputs=cropping_2d)
        model.compile(optimizer=Nadam(lr=1e-3), loss=self.jaccard_coef_loss,
                      metrics=['binary_crossentropy', self.jaccard_coef_int])
        # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # In predicting testing dataset, need to use the same mean and std in preprocessing training data
    def post_normalize_image(self, img, mean=0.338318, std=0.189734):
        img = (img - mean) / std
        return img

    def get_unet1(self):
        inputs = Input((self.crop_size, self.crop_size, 3))
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

        up6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)

        up7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)

        up8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)
        crop9 = Cropping2D(cropping=((self.edge_size, self.edge_size), (self.edge_size, self.edge_size)))(conv9)
        conv9 = BatchNormalization(axis=1)(crop9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Nadam(lr=1e-3), loss=self.jaccard_coef_loss,
                      metrics=['binary_crossentropy', self.jaccard_coef_int])
        return model

    def get_unet2(self, dropout=0.5):
        # add drop out and add one more layer

        inputs = Input((self.crop_size, self.crop_size, 3))
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
        drop5 = Dropout(dropout)(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

        conv6 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool5)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        drop6 = Dropout(dropout)(conv6)

        up7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(drop6))
        merge7 = concatenate([conv5, up7], axis=3)
        conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        drop7 = Dropout(dropout)(conv7)

        up8 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(drop7))
        merge8 = concatenate([conv4, up8], axis=3)
        conv8 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv3, up9], axis=3)

        conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)

        up10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv9))
        merge10 = concatenate([conv2, up10], axis=3)

        conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(merge10)
        conv10 = BatchNormalization(axis=1)(conv10)
        conv10 = keras.layers.advanced_activations.ELU()(conv10)
        conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv10)
        conv10 = BatchNormalization(axis=1)(conv10)
        conv10 = keras.layers.advanced_activations.ELU()(conv10)

        up11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(
            UpSampling2D(size=(2, 2))(conv10))
        merge11 = concatenate([conv1, up11], axis=3)


        crop9 = Cropping2D(cropping=((self.edge_size, self.edge_size), (self.edge_size, self.edge_size)))(conv10)
        conv9 = BatchNormalization(axis=1)(crop9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Nadam(lr=1e-3), loss=self.jaccard_coef_loss,
                      metrics=['binary_crossentropy', self.jaccard_coef_int])
        return model

    def get_model_by_name(self, model_name):
        if model_name == "unet2":
            return self.get_unet2()
        elif model_name == "unet1":
            return self.get_unet1()
        else:
            return self.get_unet0()

    def train(self, name_class, data_imageID_file, model_name, epochs=100):
        train_img = pd.read_csv(data_imageID_file)
        all_image_id = sorted(train_img.ImageId.unique())
        all_len = len(all_image_id)
        loop_time = all_len // self.get_size
        last_weight = ''
        loop_i = 0
        print(name_class)
        for i in range(loop_time):
            Image_ID = random.sample(all_image_id, self.get_size)
            all_image_id = [Image_ID2 for Image_ID2 in all_image_id if Image_ID2 not in Image_ID]
            img, msk = self.get_normalized_patches(name_class, Image_ID)
            x_trn, x_val, y_trn, y_val = train_test_split(img, msk, test_size=0.2, random_state=42)
            y_trn = y_trn[:, :, :, None]
            y_val = y_val[:, :, :, None]
            model = self.get_model_by_name(model_name)
            if i != 0:
                print("loaded")
                model.load_weights(last_weight)

            check_point_file_name = str(loop_i) + name_class + '_2.hdf5'
            model_checkpoint = ModelCheckpoint(check_point_file_name, monitor='val_jaccard_coef_int',
                                               save_best_only=True,
                                               mode='max')
            model.fit(x_trn, y_trn, batch_size=16, epochs=epochs, verbose=1, shuffle=True,
                      callbacks=[model_checkpoint],
                      validation_data=(x_val, y_val))
            last_weight = check_point_file_name
            loop_i += 1
            del x_trn, x_val, y_trn, y_val, model
            gc.collect()

        img_last = all_len - loop_time * self.get_size
        if img_last > 0:
            Image_ID = random.sample(all_image_id, img_last)

            img, msk = self.get_normalized_patches(name_class, Image_ID)
            x_trn, x_val, y_trn, y_val = train_test_split(img, msk, test_size=0.2, random_state=42)
            y_trn = y_trn[:, :, :, None]
            y_val = y_val[:, :, :, None]

            model = self.get_model_by_name(model_name)
            if loop_i != 0:
                print("loaded")
                model.load_weights(last_weight)
            check_point_file_name = str(loop_i) + '_rotate_val_jaccard_coef_int_building.hdf5'
            model_checkpoint = ModelCheckpoint(check_point_file_name, monitor='val_jaccard_coef_int',
                                               save_best_only=True,
                                               mode='max')
            model.fit(x_trn, y_trn, batch_size=16, epochs=epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint],
                      validation_data=(x_val, y_val))
