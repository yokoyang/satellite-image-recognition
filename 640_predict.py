import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import concatenate, Conv2D, Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.optimizers import Adam, Adamax, Nadam, Adadelta, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
import h5py
import keras
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
import cv2
import shapely.wkt
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
from shapely.wkt import loads as wkt_loads
from keras.backend import binary_crossentropy
import tensorflow as tf
import tifffile as tiff

from u_net_img import U_net

n_split = 4
crop_size = 160
patch_size = 288
get_size = 524
test_file_dir = '/home/yokoyang/PycharmProjects/untitled/896_val'
msk_file_dir = '/home/yokoyang/PycharmProjects/untitled/640_predict_img_result'
file_dir = test_file_dir

u_net = U_net(n_split, crop_size, patch_size, get_size, test_file_dir, msk_file_dir, file_dir)
all_class = ['countryside', 'playground', 'tree', 'road', 'building_yard', 'bare_land', 'water', 'general_building',
             'factory', 'shadow']
data_imageID_file = '/home/yokoyang/PycharmProjects/untitled/896_biaozhu/data_imageID.csv'

general_building = u_net.get_unet1()
general_building.load_weights('/home/yokoyang/PycharmProjects/untitled/model/general_building_1.hdf5')