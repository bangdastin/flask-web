from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow_probability as tfp
import traceback

# --- AWAL PERUBAHAN: Import untuk Hyperopt ---
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split

# --- AKHIR PERUBAHAN ---

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Probability version: {tfp.__version__}")

import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import pandas as pd
import os

from sklearn.metrics import average_precision_score

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    TimeDistributed,
    Lambda,
    Add,
    UpSampling2D,
    Layer,
    InputSpec,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import get_file, Progbar
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.metrics import CategoricalAccuracy

# --- Tes Minimal (Tidak Berubah) ---
try:
    print("\nMemulai tes minimal...")
    # ... (kode tes minimal Anda tetap sama) ...
    input_shape_img_test = (64, 64, 3)
    keras_input_tensor_test = Input(shape=input_shape_img_test, name="test_keras_input")
    if hasattr(keras_input_tensor_test, "_keras_history"):
        print(f"  keras_input_tensor_test memiliki _keras_history.")
    else:
        print("  keras_input_tensor_test TIDAK memiliki _keras_history.")
    output_tensor_test = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="test_standard_conv",
    )(keras_input_tensor_test)
    print(f"  SUKSES: Tes minimal - Layer Conv2D standar. Output: {output_tensor_test}")
except Exception as e:
    print(f"  ERROR tes minimal: {e}")


if K.backend() == "tensorflow":
    try:
        K.set_learning_phase(1)
        print("Keras learning phase set to 1 (training).")
    except AttributeError:
        print("K.set_learning_phase is not available.")


class Config:
    def __init__(self):
        self.verbose = True
        self.network = "vgg"
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        self.anchor_box_ratios = [[1, 1], [3, 1], [1, 3]]
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0
        self.num_rois = 32
        self.pool_size = 7
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.balanced_classes = False
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        self.class_mapping = None
        # --- AWAL PERUBAHAN: Path akan di-set di dalam objective_function ---
        self.model_path = None
        self.record_path = None
        # --- AKHIR PERUBAHAN ---
        self.base_net_weights = None
        self.use_fpn = True
        self.fpn_pyramid_levels = ["P2", "P3", "P4", "P5"]
        self.fpn_strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
        self.fpn_feature_channels = 256  # Ini akan dioptimasi
        self.anchor_box_scales = {"P2": [32], "P3": [64], "P4": [128], "P5": [256]}
        self.num_anchors_per_location = len(self.anchor_box_scales["P2"]) * len(
            self.anchor_box_ratios
        )
        # --- AWAL PERUBAHAN: Hyperparameter yang bisa dioptimasi ---
        self.learning_rate = 1e-5  # Ini akan dioptimasi
        # --- AKHIR PERUBAHAN ---


KAGGLE_BASE_INPUT_PATH = "/kaggle/input/dataset-dastin/"
KAGGLE_BASE_OUTPUT_PATH = "/kaggle/working/"
if not os.path.exists(KAGGLE_BASE_OUTPUT_PATH):
    os.makedirs(KAGGLE_BASE_OUTPUT_PATH, exist_ok=True)
TRAIN_ANNOTATION_FILENAME = "train_annotation_normalized3.txt"
VGG_WEIGHTS_FILENAME = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"


# Fungsi get_data, RoiPoolingConv, nn_base_standard, build_fpn_standard, rpn_layer_standard,
# classifier_layer, union, intersection, iou, calc_rpn, get_new_img_size, augment, get_anchor_gt
# dan fungsi-fungsi loss TETAP SAMA seperti kode terakhir Anda yang berhasil.
# Saya akan menyertakannya di akhir untuk kelengkapan, tapi tidak ada perubahan di dalamnya.
# (Untuk mempersingkat, saya akan skip definisinya di sini, tapi pastikan ada di kode Anda)
def get_data(input_path_annotation_file):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    i = 1
    dataset_image_root_path = KAGGLE_BASE_INPUT_PATH
    with open(input_path_annotation_file, "r") as f:
        print("Parsing annotation files")
        for line in f:
            if i % 50 == 0:
                sys.stdout.write(f"\ridx={i}")
                sys.stdout.flush()
            i += 1
            try:
                line_split = line.strip().split(",")
                if len(line_split) == 6:
                    (filename_relative, x1, y1, x2, y2, class_name) = line_split
                else:
                    continue
                filename_absolute = os.path.join(
                    dataset_image_root_path, filename_relative.strip()
                )
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1
                if class_name not in class_mapping:
                    if class_name == "bg" and not found_bg:
                        print("Found bg class.")
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)
                if filename_absolute not in all_imgs:
                    all_imgs[filename_absolute] = {}
                    if not os.path.exists(filename_absolute):
                        # print(f"\nWarning: Img not found {filename_absolute}. Skipping."); # Kurangi verbosity untuk HPO
                        classes_count[class_name] -= 1
                        if classes_count[class_name] == 0:
                            del classes_count[class_name]
                            continue
                        continue
                    img = cv2.imread(filename_absolute)
                    if img is None:
                        # print(f"\nWarning: Could not read {filename_absolute}. Skipping."); # Kurangi verbosity untuk HPO
                        classes_count[class_name] -= 1
                        if classes_count[class_name] == 0:
                            del classes_count[class_name]
                            continue
                        continue
                    (rows, cols) = img.shape[:2]
                    all_imgs[filename_absolute]["filepath"] = filename_absolute
                    all_imgs[filename_absolute]["width"] = cols
                    all_imgs[filename_absolute]["height"] = rows
                    all_imgs[filename_absolute]["bboxes"] = []
                all_imgs[filename_absolute]["bboxes"].append(
                    {
                        "class": class_name,
                        "x1": int(x1),
                        "x2": int(x2),
                        "y1": int(y1),
                        "y2": int(y2),
                    }
                )
            except ValueError:
                continue
            except Exception:
                continue
        sys.stdout.write("\n")
    all_data = [all_imgs[key] for key in all_imgs if all_imgs[key]]
    if (
        found_bg
        and "bg" in class_mapping
        and class_mapping["bg"] != len(class_mapping) - 1
    ):
        old_bg_idx = class_mapping["bg"]
        last_idx = len(class_mapping) - 1
        keys_at_last_idx = [
            key for key, val in class_mapping.items() if val == last_idx
        ]
        if keys_at_last_idx:
            key_to_switch = keys_at_last_idx[0]
            class_mapping["bg"] = last_idx
            class_mapping[key_to_switch] = old_bg_idx
    return all_data, classes_count, class_mapping


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        self.data_format = K.image_data_format()
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = (
            input_shape[0][3]
            if self.data_format == "channels_last"
            else input_shape[0][1]
        )
        super(RoiPoolingConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            return (
                input_shape[0][0],
                self.num_rois,
                self.nb_channels,
                self.pool_size,
                self.pool_size,
            )
        else:
            return (
                input_shape[0][0],
                self.num_rois,
                self.pool_size,
                self.pool_size,
                self.nb_channels,
            )

    def call(self, x, mask=None):
        feature_map, rois = x[0], x[1]
        batch_size = K.shape(rois)[0]
        fm_shape = K.shape(feature_map)
        if self.data_format == "channels_last":
            fm_height = K.cast(fm_shape[1], K.floatx())
            fm_width = K.cast(fm_shape[2], K.floatx())
        else:
            fm_height = K.cast(fm_shape[2], K.floatx())
            fm_width = K.cast(fm_shape[3], K.floatx())
        tf.debugging.assert_equal(
            batch_size, 1, message="RoiPoolingConv expects batch_size=1"
        )
        rois_for_crop = rois[0]
        box_indices = tf.zeros((self.num_rois), dtype=tf.int32)
        x1, y1, x2, y2 = (
            rois_for_crop[:, 0],
            rois_for_crop[:, 1],
            rois_for_crop[:, 2],
            rois_for_crop[:, 3],
        )
        y1_norm = y1 / (fm_height - 1.0 + K.epsilon())
        x1_norm = x1 / (fm_width - 1.0 + K.epsilon())
        y2_norm = y2 / (fm_height - 1.0 + K.epsilon())
        x2_norm = x2 / (fm_width - 1.0 + K.epsilon())
        normalized_boxes = K.stack([y1_norm, x1_norm, y2_norm, x2_norm], axis=1)
        feature_map_for_crop = feature_map
        if self.data_format == "channels_first":
            feature_map_for_crop = K.permute_dimensions(feature_map, (0, 2, 3, 1))
        pooled_features = tf.image.crop_and_resize(
            feature_map_for_crop,
            normalized_boxes,
            box_indices,
            (self.pool_size, self.pool_size),
            method="bilinear",
        )
        if self.data_format == "channels_first":
            pooled_features = K.permute_dimensions(pooled_features, (0, 3, 1, 2))
        reshape_dims = (
            (
                batch_size,
                self.num_rois,
                self.nb_channels,
                self.pool_size,
                self.pool_size,
            )
            if self.data_format == "channels_first"
            else (
                batch_size,
                self.num_rois,
                self.pool_size,
                self.pool_size,
                self.nb_channels,
            )
        )
        return K.reshape(pooled_features, reshape_dims)


def nn_base_standard(input_tensor_param, trainable=False):
    x = input_tensor_param
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv1",
        trainable=trainable,
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv2",
        trainable=trainable,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv1",
        trainable=trainable,
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv2",
        trainable=trainable,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    C2 = x
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv1",
        trainable=trainable,
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv2",
        trainable=trainable,
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv3",
        trainable=trainable,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    C3 = x
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv1",
        trainable=trainable,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv2",
        trainable=trainable,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv3",
        trainable=trainable,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    C4 = x
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv1",
        trainable=trainable,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv2",
        trainable=trainable,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv3",
        trainable=trainable,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool_for_c5")(x)
    C5 = x
    return input_tensor_param, {"C2": C2, "C3": C3, "C4": C4, "C5": C5}


def build_fpn_standard(backbone_feature_maps, fpn_channels=256):
    C2, C3, C4, C5 = (
        backbone_feature_maps["C2"],
        backbone_feature_maps["C3"],
        backbone_feature_maps["C4"],
        backbone_feature_maps["C5"],
    )
    P5_in = Conv2D(fpn_channels, (1, 1), padding="same", name="fpn_c5p5")(C5)
    P4_in = Conv2D(fpn_channels, (1, 1), padding="same", name="fpn_c4p4")(C4)
    P3_in = Conv2D(fpn_channels, (1, 1), padding="same", name="fpn_c3p3")(C3)
    P2_in = Conv2D(fpn_channels, (1, 1), padding="same", name="fpn_c2p2")(C2)

    def resize_like(inputs, target_tensor):
        target_shape = K.shape(target_tensor)
        if K.image_data_format() == "channels_last":
            target_h, target_w = target_shape[1], target_shape[2]
        else:
            target_h, target_w = target_shape[2], target_shape[3]
        return tf.image.resize(
            inputs, [target_h, target_w], method=tf.image.ResizeMethod.BILINEAR
        )

    def get_output_shape_for_resize_like(input_shapes):
        target_tensor_shape = input_shapes[1]
        inputs_to_resize_shape = input_shapes[0]
        if K.image_data_format() == "channels_last":
            return (
                inputs_to_resize_shape[0],
                target_tensor_shape[1],
                target_tensor_shape[2],
                inputs_to_resize_shape[3],
            )
        else:
            return (
                inputs_to_resize_shape[0],
                inputs_to_resize_shape[1],
                target_tensor_shape[2],
                target_tensor_shape[3],
            )

    P5_up = Lambda(
        lambda x: resize_like(x[0], x[1]),
        output_shape=get_output_shape_for_resize_like,
        name="fpn_p5_up",
    )([P5_in, P4_in])
    P4_td = Add(name="fpn_p4add")([P5_up, P4_in])
    P4_up = Lambda(
        lambda x: resize_like(x[0], x[1]),
        output_shape=get_output_shape_for_resize_like,
        name="fpn_p4_up",
    )([P4_td, P3_in])
    P3_td = Add(name="fpn_p3add")([P4_up, P3_in])
    P3_up = Lambda(
        lambda x: resize_like(x[0], x[1]),
        output_shape=get_output_shape_for_resize_like,
        name="fpn_p3_up",
    )([P3_td, P2_in])
    P2_td = Add(name="fpn_p2add")([P3_up, P2_in])
    P5 = Conv2D(fpn_channels, (3, 3), padding="same", name="fpn_p5")(P5_in)
    P4 = Conv2D(fpn_channels, (3, 3), padding="same", name="fpn_p4")(P4_td)
    P3 = Conv2D(fpn_channels, (3, 3), padding="same", name="fpn_p3")(P3_td)
    P2 = Conv2D(fpn_channels, (3, 3), padding="same", name="fpn_p2")(P2_td)
    return {"P2": P2, "P3": P3, "P4": P4, "P5": P5}


def rpn_layer_standard(fpn_feat, num_anchors_per_loc, level_name):
    shared = Conv2D(
        512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name=f"rpn_conv_shared_{level_name}",
    )(fpn_feat)
    x_class_logits = Conv2D(
        num_anchors_per_loc * 1,
        kernel_size=(1, 1),
        padding="same",
        activation=None,
        name=f"rpn_out_class_{level_name}_logits",
    )(shared)
    x_regr_params = Conv2D(
        num_anchors_per_loc * 4,
        kernel_size=(1, 1),
        padding="same",
        activation="linear",
        name=f"rpn_out_regress_{level_name}_params",
    )(shared)
    return x_class_logits, x_regr_params


def classifier_layer(pooled_rois, num_rois_input_shape, nb_classes_total, C_config):
    out = TimeDistributed(Flatten(name="flatten_classifier"))(pooled_rois)
    out = TimeDistributed(
        Dense(
            4096,
            activation="relu",
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
            name="fc_classifier1",
        )
    )(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(
        Dense(
            4096,
            activation="relu",
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
            name="fc_classifier2",
        )
    )(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out_class = TimeDistributed(
        Dense(
            nb_classes_total,
            activation="softmax",
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
        ),
        name=f"dense_class_{nb_classes_total}",
    )(out)
    out_regr = TimeDistributed(
        Dense(
            4 * (nb_classes_total - 1),
            activation="linear",
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
        ),
        name=f"dense_regress_{nb_classes_total}",
    )(out)
    return [out_class, out_regr]


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    return area_a + area_b - area_intersection


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    return w * h if w > 0 and h > 0 else 0


def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / (float(area_u) + 1e-6)


def calc_rpn(
    C,
    img_data,
    original_width,
    original_height,
    resized_width,
    resized_height,
    fpn_level_name,
):
    rpn_stride = C.fpn_strides[fpn_level_name]
    anchor_scales_at_level = C.anchor_box_scales[fpn_level_name]
    anchor_ratios = C.anchor_box_ratios
    num_anchor_variations = len(anchor_scales_at_level) * len(anchor_ratios)
    fh = resized_height // rpn_stride
    fw = resized_width // rpn_stride
    y_rpn_overlap = np.zeros((fh, fw, num_anchor_variations))
    y_is_box_valid = np.zeros((fh, fw, num_anchor_variations))
    y_rpn_regr = np.zeros((fh, fw, num_anchor_variations * 4))
    num_gt_bboxes = len(img_data["bboxes"])
    num_pos_anc_for_gt = np.zeros(num_gt_bboxes, dtype=int)
    best_iou_for_gt = np.zeros(num_gt_bboxes, dtype=np.float32)
    scaled_gt_bboxes = [
        {
            "x1": b["x1"] * (resized_width / float(original_width)),
            "y1": b["y1"] * (resized_height / float(original_height)),
            "x2": b["x2"] * (resized_width / float(original_width)),
            "y2": b["y2"] * (resized_height / float(original_height)),
        }
        for b in img_data["bboxes"]
    ]
    for fx_idx in range(fw):
        anc_cx = rpn_stride * (fx_idx + 0.5)
        for fy_idx in range(fh):
            anc_cy = rpn_stride * (fy_idx + 0.5)
            for scale_idx, cur_anc_scale in enumerate(anchor_scales_at_level):
                for ratio_idx, cur_anc_ratio in enumerate(anchor_ratios):
                    anc_var_idx = ratio_idx + len(anchor_ratios) * scale_idx
                    anc_w = cur_anc_scale * cur_anc_ratio[0]
                    anc_h = cur_anc_scale * cur_anc_ratio[1]
                    anc_x1, anc_y1, anc_x2, anc_y2 = (
                        anc_cx - anc_w / 2.0,
                        anc_cy - anc_h / 2.0,
                        anc_cx + anc_w / 2.0,
                        anc_cy + anc_h / 2.0,
                    )
                    if not (
                        0 <= anc_x1 < resized_width
                        and 0 <= anc_y1 < resized_height
                        and anc_x2 <= resized_width
                        and anc_y2 <= resized_height
                    ):
                        continue
                    cur_anc_bbox_type = "neg"
                    for gt_idx, gt_bbox in enumerate(scaled_gt_bboxes):
                        gt_coords = [
                            gt_bbox["x1"],
                            gt_bbox["y1"],
                            gt_bbox["x2"],
                            gt_bbox["y2"],
                        ]
                        cur_iou = iou([anc_x1, anc_y1, anc_x2, anc_y2], gt_coords)
                        if cur_iou > best_iou_for_gt[gt_idx]:
                            best_iou_for_gt[gt_idx] = cur_iou
                        if cur_iou > C.rpn_max_overlap:
                            cur_anc_bbox_type = "pos"
                            num_pos_anc_for_gt[gt_idx] += 1
                            gt_cx, gt_cy = (gt_bbox["x1"] + gt_bbox["x2"]) / 2.0, (
                                gt_bbox["y1"] + gt_bbox["y2"]
                            ) / 2.0
                            gt_w, gt_h = (
                                gt_bbox["x2"] - gt_bbox["x1"],
                                gt_bbox["y2"] - gt_bbox["y1"],
                            )
                            tx, ty, tw, th = (
                                (gt_cx - anc_cx) / anc_w,
                                (gt_cy - anc_cy) / anc_h,
                                math.log(gt_w / anc_w) if gt_w > 0 and anc_w > 0 else 0,
                                math.log(gt_h / anc_h) if gt_h > 0 and anc_h > 0 else 0,
                            )
                            y_rpn_regr[
                                fy_idx, fx_idx, anc_var_idx * 4 : anc_var_idx * 4 + 4
                            ] = [tx, ty, tw, th]
                            break
                    if (
                        C.rpn_min_overlap <= cur_iou < C.rpn_max_overlap
                        and cur_anc_bbox_type != "pos"
                    ):
                        cur_anc_bbox_type = "neutral"
                    if cur_anc_bbox_type == "pos":
                        y_is_box_valid[fy_idx, fx_idx, anc_var_idx] = 1
                        y_rpn_overlap[fy_idx, fx_idx, anc_var_idx] = 1
                    elif cur_anc_bbox_type == "neg":
                        y_is_box_valid[fy_idx, fx_idx, anc_var_idx] = 1
                        y_rpn_overlap[fy_idx, fx_idx, anc_var_idx] = 0
    for gt_idx in range(num_gt_bboxes):
        if num_pos_anc_for_gt[gt_idx] == 0 and best_iou_for_gt[gt_idx] > 0:
            gt_bbox = scaled_gt_bboxes[gt_idx]
            gt_cx, gt_cy = (gt_bbox["x1"] + gt_bbox["x2"]) / 2.0, (
                gt_bbox["y1"] + gt_bbox["y2"]
            ) / 2.0
            gt_w, gt_h = gt_bbox["x2"] - gt_bbox["x1"], gt_bbox["y2"] - gt_bbox["y1"]
            best_fx, best_fy = np.clip(
                int(round(gt_cx / rpn_stride - 0.5)), 0, fw - 1
            ), np.clip(int(round(gt_cy / rpn_stride - 0.5)), 0, fh - 1)
            anc_cx_at_fmap, anc_cy_at_fmap = rpn_stride * (
                best_fx + 0.5
            ), rpn_stride * (best_fy + 0.5)
            temp_best_iou, temp_anc_idx, temp_regr_targets = -1.0, -1, []
            for scale_idx, cur_anc_scale in enumerate(anchor_scales_at_level):
                for ratio_idx, cur_anc_ratio in enumerate(anchor_ratios):
                    anc_var_idx_loc = ratio_idx + len(anchor_ratios) * scale_idx
                    anc_w, anc_h = (
                        cur_anc_scale * cur_anc_ratio[0],
                        cur_anc_scale * cur_anc_ratio[1],
                    )
                    anc_x1, anc_y1, anc_x2, anc_y2 = (
                        anc_cx_at_fmap - anc_w / 2.0,
                        anc_cy_at_fmap - anc_h / 2.0,
                        anc_cx_at_fmap + anc_w / 2.0,
                        anc_cy_at_fmap + anc_h / 2.0,
                    )
                    if not (
                        0 <= anc_x1 < resized_width
                        and 0 <= anc_y1 < resized_height
                        and anc_x2 <= resized_width
                        and anc_y2 <= resized_height
                    ):
                        continue
                    iou_val_force = iou(
                        [anc_x1, anc_y1, anc_x2, anc_y2],
                        [gt_bbox["x1"], gt_bbox["y1"], gt_bbox["x2"], gt_bbox["y2"]],
                    )
                    if iou_val_force > temp_best_iou:
                        temp_best_iou, temp_anc_idx = iou_val_force, anc_var_idx_loc
                        tx, ty, tw, th = (
                            (gt_cx - anc_cx_at_fmap) / anc_w,
                            (gt_cy - anc_cy_at_fmap) / anc_h,
                            math.log(gt_w / anc_w) if gt_w > 0 and anc_w > 0 else 0,
                            math.log(gt_h / anc_h) if gt_h > 0 and anc_h > 0 else 0,
                        )
                        temp_regr_targets = [tx, ty, tw, th]
            if temp_anc_idx != -1:
                y_is_box_valid[best_fy, best_fx, temp_anc_idx] = 1
                y_rpn_overlap[best_fy, best_fx, temp_anc_idx] = 1
                y_rpn_regr[
                    best_fy, best_fx, temp_anc_idx * 4 : temp_anc_idx * 4 + 4
                ] = temp_regr_targets
    y_rpn_cls_target = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=2)
    y_rpn_regr_target = np.concatenate(
        [np.repeat(y_rpn_overlap, 4, axis=2), y_rpn_regr], axis=2
    )
    y_rpn_cls_target, y_rpn_regr_target = np.expand_dims(
        y_rpn_cls_target, axis=0
    ), np.expand_dims(y_rpn_regr_target, axis=0)
    num_final_pos_anc = np.sum(y_rpn_overlap)
    dummy_cls_cls, dummy_cls_regr = np.zeros((1, 1)), np.zeros((1, 4))
    return (
        np.copy(y_rpn_cls_target),
        np.copy(y_rpn_regr_target),
        num_final_pos_anc,
        dummy_cls_cls,
        dummy_cls_regr,
    )


def get_new_img_size(w, h, min_side=600):  # Pastikan fungsi ini ada atau sesuaikan
    if w <= h:
        f = float(min_side) / w if w > 0 else 0
        resized_height = int(f * h)
        resized_width = min_side
    else:
        f = float(min_side) / h if h > 0 else 0
        resized_width = int(f * w)
        resized_height = min_side
    return resized_width, resized_height


def augment(img_data, C, augment=True):  # Pastikan fungsi ini ada atau sesuaikan
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug["filepath"])
    if img is None:
        raise ValueError(f"Image not found: {img_data_aug['filepath']}")

    if augment:  # Logika augmentasi Anda
        rows, cols = img.shape[:2]
        if C.use_horizontal_flips and random.randint(0, 1) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug["bboxes"]:
                x1 = bbox["x1"]
                x2 = bbox["x2"]
                bbox["x2"] = cols - x1
                bbox["x1"] = cols - x2
        if C.use_vertical_flips and random.randint(0, 1) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug["bboxes"]:
                y1 = bbox["y1"]
                y2 = bbox["y2"]
                bbox["y2"] = rows - y1
                bbox["y1"] = rows - y2
        if C.rot_90 and random.randint(0, 1) == 0:  # ... (logika rotasi Anda)
            pass  # Implementasikan rotasi jika perlu

    img_data_aug["width"], img_data_aug["height"] = img.shape[1], img.shape[0]
    return img_data_aug, img


def get_anchor_gt(
    all_img_data, C, model_rpn_predictor=None, mode="train"
):  # Tambahkan model_rpn_predictor jika ingin RoI dari RPN
    """
    Generates batch data for Faster R-CNN training.
    - For RPN: yields RPN classification and regression targets.
    - For Classifier: yields actual RoIs and their corresponding class and regression targets.
    """
    while True:
        if mode == "train":
            random.shuffle(all_img_data)

        for img_data_orig in all_img_data:
            try:
                # Augmentasi dan pra-pemrosesan gambar (seperti sebelumnya)
                do_augment = mode == "train" and (
                    C.use_horizontal_flips or C.use_vertical_flips or C.rot_90
                )
                img_data_aug, x_img = augment(img_data_orig, C, augment=do_augment)

                original_w, original_h = img_data_aug["width"], img_data_aug["height"]
                # Gunakan get_new_img_size dari kode Anda jika berbeda, ini contoh:
                rw, rh = get_new_img_size(
                    original_w,
                    original_h,
                    min_side=C.img_min_side if hasattr(C, "img_min_side") else 600,
                )

                x_img_resized = cv2.resize(x_img, (rw, rh), interpolation=cv2.INTER_CUBIC)

                x_img_proc = x_img_resized.astype(np.float32)
                x_img_proc[:, :, 0] -= C.img_channel_mean[0]
                x_img_proc[:, :, 1] -= C.img_channel_mean[1]
                x_img_proc[:, :, 2] -= C.img_channel_mean[2]
                x_img_proc /= C.img_scaling_factor
                x_img_proc_batch = np.expand_dims(x_img_proc, axis=0)

                # 1. Generate RPN targets (seperti sebelumnya)
                Y_rpn_cls_targets_list, Y_rpn_regr_targets_list = [], []
                total_num_pos_anchors_for_image = 0
                for level_name in C.fpn_pyramid_levels:
                    y_rpn_cls_lvl, y_rpn_regr_lvl, num_pos_lvl, _, _ = calc_rpn(
                        C, img_data_aug, original_w, original_h, rw, rh, level_name
                    )
                    if y_rpn_regr_lvl.size > 0:
                        num_anc_lvl = y_rpn_regr_lvl.shape[3] // 8
                        y_rpn_regr_lvl[:, :, :, num_anc_lvl * 4 :] *= C.std_scaling

                    Y_rpn_cls_targets_list.append(y_rpn_cls_lvl)
                    Y_rpn_regr_targets_list.append(y_rpn_regr_lvl)
                    total_num_pos_anchors_for_image += num_pos_lvl

                Y_rpn_targets_combined = (
                    Y_rpn_cls_targets_list + Y_rpn_regr_targets_list
                )

                # 2. Generate actual RoIs and Classifier targets
                # Pendekatan: Gunakan GT sebagai sampel positif, dan sampel acak sebagai background
                # Ini adalah penyederhanaan; idealnya RoI berasal dari prediksi RPN.

                gt_bboxes_in_img = copy.deepcopy(
                    img_data_aug["bboxes"]
                )  # list of dicts {'class', x1,y1,x2,y2}
                num_classes = len(C.class_mapping)
                bg_class_idx = C.class_mapping["bg"]

                # Inisialisasi list untuk RoI dan targetnya
                rois_for_classifier_input = (
                    []
                )  # Akan berisi [x1, y1, x2, y2] dalam skala gambar yang di-resize (rw, rh)
                classifier_class_labels_int = []  # Label kelas integer untuk setiap RoI
                classifier_regr_targets_raw = (
                    []
                )  # Target regresi mentah (tx,ty,tw,th) untuk RoI positif
                classifier_regr_mask = []  # Mask untuk target regresi

                # Tambahkan ground truth boxes sebagai RoI positif
                for gt_box in gt_bboxes_in_img:
                    if gt_box["class"] == "bg":
                        continue  # Seharusnya tidak ada GT 'bg'

                    # Skalakan koordinat GT ke ukuran gambar yang di-resize (rw, rh)
                    x1 = gt_box["x1"] * (rw / float(original_w))
                    y1 = gt_box["y1"] * (rh / float(original_h))
                    x2 = gt_box["x2"] * (rw / float(original_w))
                    y2 = gt_box["y2"] * (rh / float(original_h))

                    rois_for_classifier_input.append([x1, y1, x2, y2])
                    classifier_class_labels_int.append(C.class_mapping[gt_box["class"]])

                    # Untuk GT box yang digunakan sebagai RoI, target regresi adalah (0,0,0,0)
                    # karena RoI sudah sempurna cocok dengan GT-nya sendiri.
                    # Nilai ini akan dinormalisasi jika diperlukan oleh loss function atau model.
                    classifier_regr_targets_raw.append([0.0, 0.0, 0.0, 0.0])

                    # Buat mask regresi
                    # Mask aktif (1.0) untuk kelas GT ini, dan 0.0 untuk kelas lain.
                    # Ukuran mask adalah 4 * (num_classes - 1)
                    current_regr_mask = np.zeros(4 * (num_classes - 1))
                    gt_class_idx_for_regr = C.class_mapping[gt_box["class"]]
                    # Pastikan kelas ini bukan 'bg' dan punya entri di array regresi
                    if (
                        gt_class_idx_for_regr < bg_class_idx
                    ):  # Asumsi 'bg' adalah kelas terakhir
                        # Indeks untuk regresi biasanya untuk kelas non-bg
                        regr_array_idx = (
                            gt_class_idx_for_regr  # Jika kelas non-bg diindeks dari 0
                        )
                        current_regr_mask[
                            regr_array_idx * 4 : regr_array_idx * 4 + 4
                        ] = 1.0
                    classifier_regr_mask.append(current_regr_mask)

                num_positive_rois = len(rois_for_classifier_input)

                # Sampel RoI background
                num_bg_rois_to_sample = C.num_rois - num_positive_rois
                if num_bg_rois_to_sample < 0:
                    num_bg_rois_to_sample = 0  # Jika sudah terlalu banyak RoI positif

                for _ in range(num_bg_rois_to_sample):
                    # Buat kotak acak, pastikan tidak tumpang tindih signifikan dengan GT
                    max_attempts = 20
                    for attempt_idx in range(max_attempts):
                        # Skala untuk lebar/tinggi RoI acak (misalnya, 10% - 50% dari dimensi gambar)
                        min_dim_roi = 0.1
                        max_dim_roi = 0.5
                        rand_w = random.uniform(min_dim_roi * rw, max_dim_roi * rw)
                        rand_h = random.uniform(min_dim_roi * rh, max_dim_roi * rh)
                        rand_x1 = random.uniform(0, rw - rand_w)
                        rand_y1 = random.uniform(0, rh - rand_h)
                        bg_roi_candidate = [
                            rand_x1,
                            rand_y1,
                            rand_x1 + rand_w,
                            rand_y1 + rand_h,
                        ]

                        is_good_bg = True
                        for gt_box in gt_bboxes_in_img:
                            gt_coords_resized = [
                                gt_box["x1"] * (rw / float(original_w)),
                                gt_box["y1"] * (rh / float(original_h)),
                                gt_box["x2"] * (rw / float(original_w)),
                                gt_box["y2"] * (rh / float(original_h)),
                            ]
                            if (
                                iou(bg_roi_candidate, gt_coords_resized)
                                > C.classifier_min_overlap
                            ):  # Threshold IoU untuk BG
                                is_good_bg = False
                                break

                        if is_good_bg or attempt_idx == max_attempts - 1:
                            rois_for_classifier_input.append(bg_roi_candidate)
                            classifier_class_labels_int.append(bg_class_idx)
                            classifier_regr_targets_raw.append(
                                [0.0, 0.0, 0.0, 0.0]
                            )  # Tidak ada regresi untuk BG
                            classifier_regr_mask.append(
                                np.zeros(4 * (num_classes - 1))
                            )  # Mask semua 0
                            break
                    if len(rois_for_classifier_input) >= C.num_rois:
                        break

                # Pastikan jumlah RoI adalah C.num_rois (padding jika perlu)
                while len(rois_for_classifier_input) < C.num_rois:
                    # Tambah RoI background dummy jika masih kurang
                    rois_for_classifier_input.append(
                        [0.0, 0.0, float(rw / 10), float(rh / 10)]
                    )  # Kotak kecil di pojok
                    classifier_class_labels_int.append(bg_class_idx)
                    classifier_regr_targets_raw.append([0.0, 0.0, 0.0, 0.0])
                    classifier_regr_mask.append(np.zeros(4 * (num_classes - 1)))

                # Potong jika terlalu banyak (seharusnya tidak terjadi dengan logika di atas)
                rois_for_classifier_input = rois_for_classifier_input[: C.num_rois]
                classifier_class_labels_int = classifier_class_labels_int[: C.num_rois]
                classifier_regr_targets_raw = classifier_regr_targets_raw[
                    : C.num_rois
                ]
                classifier_regr_mask = classifier_regr_mask[: C.num_rois]

                # Konversi ke NumPy array
                actual_rois_np = np.array(rois_for_classifier_input, dtype=np.float32)

                # RoI untuk input ke RoiPoolingConv (skala feature map P2)
                # Penting: Pastikan stride P2 benar (C.fpn_strides["P2"])
                stride_p2 = C.fpn_strides.get("P2", 4)  # Default ke 4 jika tidak ada
                rois_for_pooling_layer_np = actual_rois_np / stride_p2

                # Target kelas classifier (one-hot)
                Y_classifier_cls_target_np = np.eye(num_classes, dtype=np.float32)[
                    classifier_class_labels_int
                ]

                # Target regresi classifier (mask dan koordinat)
                # Koordinat regresi mentah (tx,ty,tw,th) perlu dinormalisasi dengan std_scaling jika loss mengharapkannya
                # Untuk GT yang digunakan sebagai RoI, targetnya [0,0,0,0], normalisasi tidak mengubahnya.
                # Jika Anda menggunakan RPN proposal, maka (tx,ty,tw,th) dihitung dan dinormalisasi.
                # classifier_regr_std = [8.0, 8.0, 4.0, 4.0] atau C.std_scaling

                # Normalisasi target regresi (jika belum)
                # Karena target mentah kita untuk GT adalah 0, normalisasi tidak berpengaruh
                # Namun, jika Anda menghitung target dari proposal RPN, Anda akan melakukan:
                # normalized_regr_targets = raw_regr_targets / C.classifier_regr_std (elemen-wise)
                # Untuk sekarang, classifier_regr_targets_raw sudah [0,0,0,0] untuk positif, jadi kita biarkan.

                final_regr_targets_raw_np = np.array(
                    classifier_regr_targets_raw, dtype=np.float32
                )
                final_regr_mask_np = np.array(classifier_regr_mask, dtype=np.float32)

                # Bentuk akhir target regresi untuk loss function: (batch, num_rois, 4 * (num_classes - 1) * 2)
                # Di mana bagian pertama adalah mask, bagian kedua adalah koordinat.
                Y_classifier_regr_target_masked_np = np.concatenate(
                    [final_regr_mask_np, final_regr_targets_raw_np], axis=1
                )

                Y_classifier_targets_actual = [
                    np.expand_dims(
                        Y_classifier_cls_target_np, axis=0
                    ),  # (1, num_rois, num_classes)
                    np.expand_dims(
                        Y_classifier_regr_target_masked_np, axis=0
                    ),  # (1, num_rois, (4*(num_classes-1))*2)
                ]

                yield (
                    x_img_proc_batch,  # Input gambar (batch_size=1)
                    np.expand_dims(
                        rois_for_pooling_layer_np, axis=0
                    ),  # RoI untuk pooling (batch_size=1, num_rois, 4)
                    Y_rpn_targets_combined,  # Target RPN (list of arrays)
                    Y_classifier_targets_actual,  # Target Classifier aktual ([cls], [regr_mask_coords])
                    img_data_aug,  # Info gambar (untuk debug atau evaluasi)
                    total_num_pos_anchors_for_image,  # Jumlah anchor positif RPN
                )

            except Exception as e_generator:
                print(
                    f"Error dalam get_anchor_gt untuk gambar {img_data_orig.get('filepath','N/A')}: {e_generator}"
                )
                traceback.print_exc()
                continue


lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0
lambda_cls_regr = 1.0
lambda_cls_class = 1.0


def rpn_loss_cls_bayesian(num_anchors_at_loc):
    def rpn_loss_cls_fixed_num(y_true, y_pred_logits):
        y_is_box_valid = tf.cast(y_true[..., :num_anchors_at_loc], tf.float32)
        y_true_labels = tf.cast(y_true[..., num_anchors_at_loc:], tf.float32)
        log_likelihood = tfp.distributions.Bernoulli(logits=y_pred_logits).log_prob(
            y_true_labels
        )
        masked_log_likelihood = log_likelihood * y_is_box_valid
        return (
            lambda_rpn_class
            * -tf.reduce_sum(masked_log_likelihood)
            / (tf.reduce_sum(y_is_box_valid) + K.epsilon())
        )

    return rpn_loss_cls_fixed_num


def rpn_loss_regr_bayesian(num_anchors_at_loc, regression_sigma=1.0):
    def rpn_loss_regr_fixed_num(y_true, y_pred_means):
        mask = tf.cast(y_true[..., : num_anchors_at_loc * 4], tf.float32)
        y_true_targets = y_true[..., num_anchors_at_loc * 4 :]
        log_likelihood = tfp.distributions.Normal(
            loc=y_pred_means, scale=regression_sigma
        ).log_prob(y_true_targets)
        masked_log_likelihood = log_likelihood * mask
        return (
            lambda_rpn_regr
            * -tf.reduce_sum(masked_log_likelihood)
            / (tf.reduce_sum(mask) + K.epsilon())
        )

    return rpn_loss_regr_fixed_num


def class_loss_cls_bayesian(y_true_one_hot, y_pred_logits):
    y_true_sel = y_true_one_hot[0, :, :]
    y_pred_logits_sel = y_pred_logits[0, :, :]
    log_likelihood = tfp.distributions.OneHotCategorical(
        logits=y_pred_logits_sel, dtype=tf.float32
    ).log_prob(tf.cast(y_true_sel, tf.float32))
    return lambda_cls_class * -tf.reduce_mean(log_likelihood)


def class_loss_regr_bayesian(num_classes_inc_bg, regression_sigma=1.0):
    def class_loss_regr_fixed_num(y_true, y_pred_means):
        num_cls_regr = num_classes_inc_bg - 1
        mask = y_true[..., : 4 * num_cls_regr]
        true_coords = y_true[..., 4 * num_cls_regr :]
        log_likelihood = tfp.distributions.Normal(
            loc=y_pred_means, scale=regression_sigma
        ).log_prob(true_coords)
        masked_log_likelihood = log_likelihood * mask
        return (
            lambda_cls_regr
            * -K.sum(masked_log_likelihood)
            / (K.sum(mask) + K.epsilon())
        )

    return class_loss_regr_fixed_num


train_path = os.path.join(KAGGLE_BASE_INPUT_PATH, TRAIN_ANNOTATION_FILENAME)
output_weight_path = os.path.join(
    KAGGLE_BASE_OUTPUT_PATH, "model_frcnn_fpn_standard_kaggle.weights.h5"
)
record_path = os.path.join(KAGGLE_BASE_OUTPUT_PATH, "record_fpn_standard_kaggle.csv")
base_weight_path_vgg = os.path.join(KAGGLE_BASE_INPUT_PATH, VGG_WEIGHTS_FILENAME)
config_output_filename = os.path.join(
    KAGGLE_BASE_OUTPUT_PATH, "config_fpn_standard_kaggle.pickle"
)
YOUR_TOTAL_TRAINING_SAMPLES = 1018
C = Config()
C.model_path = output_weight_path
C.base_net_weights = base_weight_path_vgg
C.record_path = record_path
print(f"Path anotasi training: {train_path}, Ada: {os.path.exists(train_path)}")
print(
    f"Path bobot VGG dasar: {C.base_net_weights}, Ada: {os.path.exists(C.base_net_weights)}"
)
st = time.time()
train_imgs, classes_count, class_mapping = get_data(train_path)
sys.stdout.write("\n")
print(f"Menghabiskan {(time.time()-st)/60:.2f} menit memuat data")
if not train_imgs:
    raise ValueError("Tidak ada data training.")
if "bg" not in classes_count:
    classes_count["bg"] = 0
if "bg" not in class_mapping:
    class_mapping["bg"] = len(class_mapping)
if "bg" in class_mapping and class_mapping["bg"] != len(class_mapping) - 1:
    old_bg_idx = class_mapping["bg"]
    last_idx = len(class_mapping) - 1
    keys_at_last_idx = [k for k, v in class_mapping.items() if v == last_idx]
    if keys_at_last_idx:
        key_to_switch = keys_at_last_idx[0]
        class_mapping[key_to_switch] = old_bg_idx
        class_mapping["bg"] = last_idx
C.class_mapping = class_mapping
print("Kelas training:")
pprint.pprint(classes_count)
nb_classes = len(classes_count)
print(f"Jumlah kelas (inc. bg) = {nb_classes}")
print("Class mapping:", class_mapping)
with open(config_output_filename, "wb") as config_f:
    pickle.dump(C, config_f)
    print(f"Config disimpan ke {config_output_filename}")
random.seed(1)
random.shuffle(train_imgs)
print(f"Jumlah sampel training: {len(train_imgs)}")
data_gen_train = get_anchor_gt(train_imgs, C, mode="train")

if K.image_data_format() == "channels_first":
    input_shape_img_model = (3, None, None)
else:
    input_shape_img_model = (None, None, 3)
img_input = Input(shape=input_shape_img_model, name="image_input_main")
roi_input_for_model = Input(shape=(C.num_rois, 4), name="roi_input_model_all")
backbone_input_tensor, backbone_feature_maps_dict = nn_base_standard(
    img_input, trainable=True
)
fpn_outputs_dict = build_fpn_standard(
    backbone_feature_maps_dict, fpn_channels=C.fpn_feature_channels
)
(
    rpn_all_cls_outputs,
    rpn_all_regr_outputs,
    rpn_all_losses_cls_fns,
    rpn_all_losses_regr_fns,
) = ([], [], [], [])
for level_name in C.fpn_pyramid_levels:
    fpn_level_feat = fpn_outputs_dict[level_name]
    level_cls_logits, level_regr_params = rpn_layer_standard(
        fpn_level_feat, C.num_anchors_per_location, level_name
    )
    rpn_all_cls_outputs.append(level_cls_logits)
    rpn_all_regr_outputs.append(level_regr_params)
    rpn_all_losses_cls_fns.append(rpn_loss_cls_bayesian(C.num_anchors_per_location))
    rpn_all_losses_regr_fns.append(
        rpn_loss_regr_bayesian(C.num_anchors_per_location, regression_sigma=1.0)
    )
model_rpn_outputs_combined = rpn_all_cls_outputs + rpn_all_regr_outputs
pooled_rois = RoiPoolingConv(C.pool_size, C.num_rois, name="roi_pooling_conv_main")(
    [fpn_outputs_dict["P2"], roi_input_for_model]
)
classifier_outputs_list = classifier_layer(pooled_rois, C.num_rois, nb_classes, C)
model_rpn = Model(
    inputs=img_input, outputs=model_rpn_outputs_combined, name="rpn_standard_fpn"
)
model_all = Model(
    inputs=[img_input, roi_input_for_model],
    outputs=model_rpn_outputs_combined + classifier_outputs_list,
    name="faster_rcnn_fpn_standard",
)
optimizer = Adam(learning_rate=1e-5)
try:
    model_rpn_losses_compile_list = rpn_all_losses_cls_fns + rpn_all_losses_regr_fns
    model_rpn.compile(optimizer=optimizer, loss=model_rpn_losses_compile_list)
    print("Model RPN (Standard FPN) berhasil dikompilasi.")
except Exception as e:
    print(f"Error kompilasi model_rpn: {e}")
model_all_losses_compile_list = (
    rpn_all_losses_cls_fns
    + rpn_all_losses_regr_fns
    + [
        class_loss_cls_bayesian,
        class_loss_regr_bayesian(nb_classes, regression_sigma=1.0),
    ]
)
model_all_loss_weights_list = [1.0] * len(model_all_losses_compile_list)
try:
    # Tentukan nama output classifier_cls (sesuai dengan penamaan di classifier_layer)
    # classifier_outputs_list[0] adalah out_class
    classifier_cls_output_layer_name = f"dense_class_{nb_classes}"

    metrics_map = {
        classifier_cls_output_layer_name: CategoricalAccuracy(name='classifier_accuracy')
    }

    model_all.compile(
        optimizer=optimizer,
        loss=model_all_losses_compile_list,
        loss_weights=model_all_loss_weights_list,
        metrics=metrics_map, # <--- TAMBAHKAN METRIK DI SINI
        jit_compile=False
    )
    print("Model Keseluruhan (Standard Faster R-CNN FPN) berhasil dikompilasi dengan metrik akurasi.")
    print(f"Output names: {[output.name for output in model_all.outputs]}") # Debugging output names
    print(f"Model metrics names: {model_all.metrics_names}") 
except Exception as e:
    print(f"Error kompilasi model_all: {e}")
    traceback.print_exc() # Cetak traceback lengkap untuk error kompilasi

record_df = pd.DataFrame()
if os.path.exists(C.model_path):
    print(f"Mencoba melanjutkan training dari bobot: {C.model_path}")
    try:
        model_all.load_weights(C.model_path, by_name=True)
        print("Bobot model_all berhasil dimuat.")
    except Exception as e_load_all:
        print(f"Error memuat bobot model_all: {e_load_all}. Mencoba VGG dasar.")
        if C.base_net_weights and os.path.exists(C.base_net_weights):
            try:
                _, temp_backbone_outputs_dict = nn_base_standard(
                    img_input, trainable=False
                )
                temp_outputs_list_for_loading = [
                    temp_backbone_outputs_dict[level]
                    for level in ["C2", "C3", "C4", "C5"]
                ]
                temp_backbone_model_for_loading = Model(
                    inputs=img_input, outputs=temp_outputs_list_for_loading
                )
                temp_backbone_model_for_loading.load_weights(
                    C.base_net_weights, by_name=True
                )
                print(f"Bobot VGG dasar dari {C.base_net_weights} berhasil dimuat.")
            except Exception as e_vgg:
                print(f"Error memuat bobot VGG dasar: {e_vgg}")
        else:
            print("Bobot VGG dasar tidak ada/tidak dispesifikasi.")
    if os.path.exists(C.record_path):
        record_df = pd.read_csv(C.record_path)
        print(f"Sudah terlatih {len(record_df)} iterasi.")
else:
    print("Training pertama (Standard FPN).")
    if C.base_net_weights and os.path.exists(C.base_net_weights):
        try:
            _, temp_backbone_outputs_dict = nn_base_standard(img_input, trainable=False)
            temp_outputs_list_for_loading = [
                temp_backbone_outputs_dict[level] for level in ["C2", "C3", "C4", "C5"]
            ]
            temp_backbone_model_for_loading = Model(
                inputs=img_input, outputs=temp_outputs_list_for_loading
            )
            temp_backbone_model_for_loading.load_weights(
                C.base_net_weights, by_name=True
            )
            print(f"Bobot VGG dasar dari {C.base_net_weights} berhasil dimuat.")
        except Exception as e_vgg:
            print(f"Gagal memuat VGG dasar: {e_vgg}. Backbone dilatih dari awal.")
    else:
        print("Bobot VGG dasar tidak ada. Backbone dilatih dari awal.")
if record_df.empty:
    record_df = pd.DataFrame(
        columns=[
            "mean_overlapping_bboxes",
            "classifier_acc", # <--- NAMA KOLOM BARU UNTUK AKURASI
            "loss_rpn_cls",
            "loss_rpn_regr",
            "loss_cls_cls",
            "loss_cls_regr",
            # "kl_divergence_rpn", # Ini sudah Anda hapus/ganti
            # "kl_divergence_cls", # Ini sudah Anda hapus/ganti
            "kl_divergence_total", # Kolom yang sudah ada
            # "curr_loss_elbo", # Ini sudah Anda ganti
            "total_objective_loss", # Kolom yang sudah ada
            "elapsed_time",
            "mAP",
        ]
    )
num_new_epochs = 200
epoch_length = 32
total_epochs_existing = len(record_df) if not record_df.empty else 0
r_epochs_completed = total_epochs_existing
total_epochs_planned = total_epochs_existing + num_new_epochs
best_loss_nll = np.Inf

# Jika kolom 'class_acc' lama masih ada dan ingin diganti namanya
if 'class_acc' in record_df.columns and 'classifier_acc' not in record_df.columns:
    record_df = record_df.rename(columns={'class_acc': 'classifier_acc'}, errors='ignore')
elif 'class_acc' in record_df.columns and 'classifier_acc' in record_df.columns and 'class_acc' != 'classifier_acc':
    # Jika keduanya ada dan berbeda, mungkin hapus yang lama jika tidak terpakai
    record_df = record_df.drop(columns=['class_acc'], errors='ignore')
    
if "curr_loss_elbo" in record_df.columns and not record_df["curr_loss_elbo"].empty:
    valid_losses = record_df["curr_loss_elbo"].dropna()
    if not valid_losses.empty:
        best_loss_nll = valid_losses.min()
print(
    f"Training untuk {num_new_epochs} epoch baru. Panjang epoch: {epoch_length} iterasi."
)
print(f"Best loss NLL (sebelumnya ELBO) sebelumnya: {best_loss_nll}")
# epoch_loss_tracker = np.zeros((epoch_length, 8)) # Akan diinisialisasi di dalam loop epoch
print("Memulai training loop (model standar)...")
training_start_time_global = time.time()

for epoch_idx_new in range(num_new_epochs):
    current_epoch_total = r_epochs_completed + 1
    progbar = Progbar(epoch_length, verbose=1, stateful_metrics=["Total_Loss_Obj"])
    print(f"Epoch {current_epoch_total}/{total_epochs_planned}")
    epoch_start_time = time.time()
    iter_in_epoch = 0

    # Kolom: [rpn_cls, rpn_regr, cls_cls, cls_regr, kl_total, total_obj_loss, cls_acc]
    num_loss_tracker_cols = 7 
    epoch_loss_tracker = np.zeros((epoch_length, num_loss_tracker_cols))

    X_batch_iter, X_rois_iter, Y_rpn_targets_iter, Y_classifier_targets_iter = (
        None,
        None,
        None,
        None,
    )

    while iter_in_epoch < epoch_length:
        try:
            (
                X_batch_iter,
                X_rois_iter,
                Y_rpn_targets_iter,
                Y_classifier_targets_iter,
                img_data_iter,
                current_num_pos_rpn_anchors,
            ) = next(
                data_gen_train
            )

            if iter_in_epoch == 0 and current_epoch_total == 1: # Debugging shapes
                print("\n--- Debugging Shapes KONKRET (Epoch {current_epoch_total}, Iterasi 0) ---")
                print(f"Bentuk X_batch_iter (gambar): {X_batch_iter.shape}")
                print(f"Bentuk X_rois_iter (RoI input): {X_rois_iter.shape}")
                num_fpn_levels = len(C.fpn_pyramid_levels)
                print("\nTarget Shapes RPN (Y_rpn_targets_iter):")
                if isinstance(Y_rpn_targets_iter, list) and len(Y_rpn_targets_iter) == 2 * num_fpn_levels:
                    for i_level in range(num_fpn_levels):
                        print(f"  Target Cls RPN P{i_level+2} shape: {Y_rpn_targets_iter[i_level].shape}")
                    for i_level in range(num_fpn_levels):
                        print(f"  Target Regr RPN P{i_level+2} shape: {Y_rpn_targets_iter[num_fpn_levels + i_level].shape}")
                else:
                    print(f"  Format Y_rpn_targets_iter tidak terduga atau num_fpn_levels salah.")

                print("\nClassifier Target Shapes (Y_classifier_targets_iter):")
                if isinstance(Y_classifier_targets_iter, list) and len(Y_classifier_targets_iter) == 2:
                    print(f"  Target Cls Classifier shape: {Y_classifier_targets_iter[0].shape}")
                    print(f"  Target Regr Classifier shape: {Y_classifier_targets_iter[1].shape}")
                else:
                    print(f"  Format Y_classifier_targets_iter tidak terduga.")
                print("--- Akhir Debugging Shapes KONKRET ---\n")

            all_targets_for_model_all = Y_rpn_targets_iter + Y_classifier_targets_iter

            metrics_all_batch = model_all.train_on_batch(
                [X_batch_iter, X_rois_iter],
                all_targets_for_model_all
            )

            kl_divergence_total_batch_value = sum(
                K.get_value(loss) for loss in model_all.losses
            )

            num_fpn_levels = len(C.fpn_pyramid_levels)
            num_rpn_outputs = 2 * num_fpn_levels
            num_classifier_outputs = 2
            expected_num_individual_losses = num_rpn_outputs + num_classifier_outputs

            total_nll_val_from_metrics = 0.0
            avg_rpn_cls_loss_b = 0.0
            avg_rpn_regr_loss_b = 0.0
            classifier_cls_loss_b = 0.0
            classifier_regr_loss_b = 0.0

            if isinstance(metrics_all_batch, (list, np.ndarray)):
                metrics_list = list(metrics_all_batch)
                offset = 0
                if len(metrics_list) == 1 + expected_num_individual_losses: # total_loss + individual_losses
                    total_nll_val_from_metrics = metrics_list[0]
                    offset = 1
                elif len(metrics_list) == expected_num_individual_losses: # individual_losses only
                    total_nll_val_from_metrics = sum(metrics_list)
                else:
                    print(f"Peringatan: Format metrics_all_batch tidak terduga (panjang {len(metrics_list)}). Menggunakan nilai pertama jika ada.")
                    total_nll_val_from_metrics = metrics_list[0] if len(metrics_list) > 0 else 0.0

                if len(metrics_list) >= offset + expected_num_individual_losses:
                    rpn_cls_losses = metrics_list[offset : offset + num_fpn_levels]
                    rpn_regr_losses = metrics_list[
                        offset + num_fpn_levels : offset + num_rpn_outputs
                    ]
                    classifier_cls_loss_b = metrics_list[offset + num_rpn_outputs]
                    classifier_regr_loss_b = metrics_list[
                        offset + num_rpn_outputs + 1
                    ]

                    avg_rpn_cls_loss_b = np.mean(rpn_cls_losses) if len(rpn_cls_losses) > 0 else 0.0
                    avg_rpn_regr_loss_b = np.mean(rpn_regr_losses) if len(rpn_regr_losses) > 0 else 0.0
            elif isinstance(metrics_all_batch, (float, np.float32, np.float64)):
                total_nll_val_from_metrics = metrics_all_batch
            else:
                print(f"Peringatan: Tipe metrics_all_batch tidak terduga: {type(metrics_all_batch)}")

            current_total_objective_loss = (
                total_nll_val_from_metrics + kl_divergence_total_batch_value
            )

            # Dapatkan nilai akurasi classifier
            # Nama metriknya harus sesuai dengan yang ada di model_all.metrics_names
            # Biasanya akan menjadi 'classifier_accuracy' atau nama_output + '_' + nama_metrik
            classifier_acc_value = 0.0 # Default jika tidak ditemukan
            # Cari nama metrik akurasi yang benar dari model.metrics_names
            # Setelah kompilasi, kita print model_all.metrics_names.
            # Misalkan nama metriknya adalah 'classifier_accuracy' (karena kita menamakannya demikian)
            try:
                acc_metric_name_in_list = 'classifier_accuracy' 
                if acc_metric_name_in_list in model_all.metrics_names:
                    acc_index = model_all.metrics_names.index(acc_metric_name_in_list)
                    classifier_acc_value = metrics_all_batch[acc_index]
                else:
                    # Fallback
                    if len(metrics_all_batch) > (1 + expected_num_individual_losses):
                        classifier_acc_value = metrics_all_batch[1 + expected_num_individual_losses] # Atau metrics_all_batch[-1] jika itu satu-satunya metrik
                    else: # Tambahkan else di sini
                        print(f"Peringatan: Metrik '{acc_metric_name_in_list}' tidak ditemukan dan fallback juga gagal.")
                        classifier_acc_value = 0.0

            except (ValueError, IndexError) as e_acc:
                print(f"Peringatan: Tidak dapat mengambil classifier_accuracy dari metrics_all_batch. Error: {e_acc}")
                print(f"  metrics_all_batch: {metrics_all_batch}")
                print(f"  model_all.metrics_names: {model_all.metrics_names}")
                classifier_acc_value = 0.0 # Atau np.nan

            epoch_loss_tracker[iter_in_epoch, 0] = avg_rpn_cls_loss_b
            epoch_loss_tracker[iter_in_epoch, 1] = avg_rpn_regr_loss_b
            epoch_loss_tracker[iter_in_epoch, 2] = classifier_cls_loss_b
            epoch_loss_tracker[iter_in_epoch, 3] = classifier_regr_loss_b
            epoch_loss_tracker[iter_in_epoch, 4] = kl_divergence_total_batch_value
            epoch_loss_tracker[iter_in_epoch, 5] = current_total_objective_loss
            epoch_loss_tracker[iter_in_epoch, 6] = classifier_acc_value # <-- SIMPAN AKURASI

            progbar_values = [
                ("RPN_Cls", avg_rpn_cls_loss_b),
                ("RPN_Regr", avg_rpn_regr_loss_b),
                ("CLS_Cls", classifier_cls_loss_b),
                ("CLS_Regr", classifier_regr_loss_b),
                ("KL_Tot", kl_divergence_total_batch_value),
                ("Total_Loss_Obj", current_total_objective_loss),
                ("Cls_Acc", classifier_acc_value), # <--- TAMBAHKAN KE PROGBAR
            ]
            progbar.update(iter_in_epoch + 1, values=progbar_values)
            iter_in_epoch += 1

        except StopIteration:
            print("Data generator selesai lebih awal untuk epoch ini.")
            epoch_length = iter_in_epoch # Perbarui epoch_length jika generator berhenti lebih awal
            break 
        except Exception as e_train_iter:
            print(
                f"Error pada iterasi {iter_in_epoch} di epoch {current_epoch_total}: {e_train_iter}"
            )
            traceback.print_exc()
            epoch_length = iter_in_epoch # Perbarui epoch_length jika terjadi error
            break 
    
    # --- Setelah loop iterasi selesai untuk epoch ini ---
    if epoch_length > 0: # Hanya hitung rata-rata jika ada iterasi yang selesai
        avg_loss_rpn_cls_epoch = np.mean(epoch_loss_tracker[:epoch_length, 0])
        avg_loss_rpn_regr_epoch = np.mean(epoch_loss_tracker[:epoch_length, 1])
        avg_loss_cls_cls_epoch = np.mean(epoch_loss_tracker[:epoch_length, 2])
        avg_loss_cls_regr_epoch = np.mean(epoch_loss_tracker[:epoch_length, 3])
        avg_kl_total_epoch = np.mean(epoch_loss_tracker[:epoch_length, 4])
        avg_total_objective_loss_epoch = np.mean(epoch_loss_tracker[:epoch_length, 5])
        avg_classifier_acc_epoch = np.mean(epoch_loss_tracker[:epoch_length, 6]) # <--- HITUNG RATA-RATA AKURASI
    else: 
        avg_loss_rpn_cls_epoch, avg_loss_rpn_regr_epoch = 0, 0
        avg_loss_cls_cls_epoch, avg_loss_cls_regr_epoch = 0, 0
        avg_kl_total_epoch = 0
        avg_total_objective_loss_epoch = np.Inf
        avg_classifier_acc_epoch = 0.0 #


    epoch_duration_sec = time.time() - epoch_start_time
    epoch_duration_min = epoch_duration_sec / 60.0

    print(f"\nEpoch {current_epoch_total} Selesai. Durasi: {epoch_duration_min:.2f} min.")
    print(
        f"  Rata-rata Loss RPN Cls: {avg_loss_rpn_cls_epoch:.4f}, Regr: {avg_loss_rpn_regr_epoch:.4f}"
    )
    print(
        f"  Rata-rata Loss Classifier Cls: {avg_loss_cls_cls_epoch:.4f}, Regr: {avg_loss_cls_regr_epoch:.4f}"
    )
    print(f"  Rata-rata KL Total: {avg_kl_total_epoch:.5f}")
    print(
        f"  Rata-rata Total Objective Loss (NLL+KL): {avg_total_objective_loss_epoch:.4f}"
    )
    print(f"  Rata-rata Classifier Accuracy: {avg_classifier_acc_epoch:.4f}")

    if avg_total_objective_loss_epoch < best_loss_nll:
        print(
            f"  Total Objective Loss menurun dari {best_loss_nll:.4f} ke {avg_total_objective_loss_epoch:.4f}, menyimpan bobot..."
        )
        best_loss_nll = avg_total_objective_loss_epoch
        model_all.save_weights(C.model_path)

    new_row_data = {
        "mean_overlapping_bboxes": 0, 
        "classifier_acc": round(avg_classifier_acc_epoch, 4), # <--- GUNAKAN NAMA BARU DAN NILAI BARU
        "loss_rpn_cls": round(avg_loss_rpn_cls_epoch, 4),
        "loss_rpn_regr": round(avg_loss_rpn_regr_epoch, 4),
        "loss_cls_cls": round(avg_loss_cls_cls_epoch, 4),
        "loss_cls_regr": round(avg_loss_cls_regr_epoch, 4),
        "kl_divergence_total": round(avg_kl_total_epoch, 5),
        "total_objective_loss": round(avg_total_objective_loss_epoch, 5),
        "elapsed_time": round(epoch_duration_min, 2),
        "mAP": 0, # mAP masih placeholder
    }

    if "curr_loss_elbo" in record_df.columns:
        record_df = record_df.rename(
            columns={"curr_loss_elbo": "total_objective_loss"}, errors="ignore"
        )
    if "kl_divergence_rpn" in record_df.columns: # Hapus kolom lama jika ada
        record_df = record_df.drop(
            columns=["kl_divergence_rpn", "kl_divergence_cls"], errors="ignore"
        )
        
    record_df = pd.concat([record_df, pd.DataFrame([new_row_data])], ignore_index=True)
    record_df.to_csv(C.record_path, index=False)
    r_epochs_completed += 1
    
    if epoch_length == 0: # Jika epoch berhenti karena error/StopIteration di awal
        print("Epoch dihentikan karena tidak ada iterasi yang selesai.")
        break # Hentikan loop epoch utama


total_training_duration_min = (time.time() - training_start_time_global) / 60
print(f"Training selesai. Total waktu: {total_training_duration_min:.2f} menit.")
if not record_df.empty:
    plt.figure(figsize=(18, 12))
    # Menggunakan nama kolom yang sudah diperbarui di record_df
    plot_titles = [
        "Loss RPN Cls",
        "Loss RPN Regr",
        "Loss Classifier Cls",
        "Loss Classifier Regr",
        "KL Divergence Total",
        "Total Objective Loss",
        "Classifier Accuracy", # <--- TAMBAHKAN JUDUL PLOT AKURASI
        "Elapsed Time (min)",
    ]
    plot_columns = [
        "loss_rpn_cls",
        "loss_rpn_regr",
        "loss_cls_cls",
        "loss_cls_regr",
        "kl_divergence_total", 
        "total_objective_loss", 
        "classifier_acc", # <--- TAMBAHKAN KOLOM AKURASI
        "elapsed_time",
    ]
    
    num_plots = len(plot_columns)
    # Sesuaikan layout subplot jika jumlah plot berubah
    # Misalnya, 3x3 jika ada 7-9 plot
    num_rows_plot = math.ceil(num_plots / 3)

    for i, (title, column) in enumerate(zip(plot_titles, plot_columns)):
        plt.subplot(num_rows_plot, 3, i + 1)
        if column in record_df.columns:
            plt.plot(record_df[column])
            plt.title(title)
            plt.xlabel("Epoch") # Mengganti "Iterasi" menjadi "Epoch" karena data per epoch
            plt.ylabel("Nilai")
        else:
            plt.title(f"{title} (Data tidak ada)")
    plt.tight_layout()
    plot_output_path = os.path.join(
        KAGGLE_BASE_OUTPUT_PATH, "training_plots_fpn_standard_v2.png" # Nama file plot baru
    )
    plt.savefig(plot_output_path)
    print(f"Plot disimpan ke: {plot_output_path}")
    plt.show()
else:
    print("Record_df kosong. Tidak ada plot yang dihasilkan.")
