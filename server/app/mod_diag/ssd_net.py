from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from .models.keras_ssd300 import ssd_300
from .keras_loss_function.keras_ssd_loss import SSDLoss
from .keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from .keras_layers.keras_layer_L2Normalization import L2Normalization
from .ssd_box_utils.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from .settings300 import *
import numpy as np


class SSD_net(object):

    def __init__(self, model_path, weights_path):

        self.model_path = model_path
        self.weights_path = weights_path
        self.model = ''

    def get_ssd(self):

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
        model = load_model(self.model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                       'L2Normalization': L2Normalization,
                                                       'compute_loss': ssd_loss.compute_loss})
        model.load_weights(self.weights_path, by_name=True)

        predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                           model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

        ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        min_scale=None,
                                        max_scale=None,
                                        scales=scales,
                                        aspect_ratios_global=None,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        limit_boxes=limit_boxes,
                                        variances=variances,
                                        pos_iou_threshold=0.5,
                                        neg_iou_threshold=0.2,
                                        coords=coords,
                                        normalize_coords=normalize_coords)

        self.model = model


    def predict(self, image):

        img = image.resize((300, 300))
        X = np.array([np.array(img)])
        y_pred = self.model.predict(X)
        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=0.2,
                                  iou_threshold=0.2,
                                  top_k=1,
                                  input_coords='centroids',
                                  normalize_coords=normalize_coords,
                                  img_height=img_height,
                                  img_width=img_width)

        box_list = []
        for box in y_pred_decoded[0]:
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            box_list.append([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

        return box_list
