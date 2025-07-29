# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import cv2
import numpy as np
import pickle
import random
import time
import math
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten,
    Dense, Dropout, Layer, Add, UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# -------------------------------------------------------
# KONFIGURASI APLIKASI
# -------------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)
K.set_image_data_format('channels_last')
print("Sesi TensorFlow siap (menggunakan default TF 2.x).")

# --- Path ke Model dan Konfigurasi ---
MODEL_PATH = "model_frcnn_fpn_standard_kaggle.weights.h5"
CONFIG_PATH = "config_fpn_standard_kaggle.pickle"
SCORE_THRESH = 0.7

# --------------------------------------------------------------------
# BAGIAN 1: DEFINISI KELAS DAN FUNGSI (Diambil dari skrip training)
# --------------------------------------------------------------------

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
        self.model_path = None
        self.record_path = None
        self.base_net_weights = None
        self.use_fpn = True
        self.fpn_pyramid_levels = ["P2", "P3", "P4", "P5"]
        self.fpn_strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
        self.fpn_feature_channels = 256
        self.anchor_box_scales = {"P2": [32], "P3": [64], "P4": [128], "P5": [256]}
        self.num_anchors_per_location = len(self.anchor_box_scales.get("P2", [])) * len(
            self.anchor_box_ratios
        )
        self.learning_rate = 1e-5

class RoiPoolingConv(Layer):
    """Lapisan ROI Pooling untuk FPN."""
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        feature_map, rois = x[0], x[1]
        batch_size = K.shape(rois)[0]
        num_rois_here = K.shape(rois)[1]
        rois_flat = K.reshape(rois, (-1, 4))
        box_indices = tf.repeat(tf.range(batch_size), repeats=num_rois_here)

        fm_shape = K.shape(feature_map)
        fm_height = K.cast(fm_shape[1], K.floatx())
        fm_width = K.cast(fm_shape[2], K.floatx())

        y1 = rois_flat[:, 1] / (fm_height - 1.0)
        x1 = rois_flat[:, 0] / (fm_width - 1.0)
        y2 = rois_flat[:, 3] / (fm_height - 1.0)
        x2 = rois_flat[:, 2] / (fm_width - 1.0)

        normalized_boxes = K.stack([y1, x1, y2, x2], axis=1)

        pooled_features = tf.image.crop_and_resize(
            feature_map, normalized_boxes, box_indices, (self.pool_size, self.pool_size)
        )
        final_output = K.reshape(pooled_features, (batch_size, num_rois_here, self.pool_size, self.pool_size, self.nb_channels))
        return final_output

def apply_regr(x, y, w, h, tx, ty, tw, th):
    """Menerapkan regresi bounding box."""
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        return x1, y1, w1, h1
    except (ValueError, OverflowError):
        return x, y, w, h

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return [], []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    pick = []
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    if len(pick) > max_boxes: pick = pick[:max_boxes]
    return boxes[pick], probs[pick]

def rpn_to_roi_fpn(rpn_cls_outputs, rpn_regr_outputs, C, img_shape):
    """Mengkonversi output RPN dari semua level FPN menjadi RoI."""
    all_proposals = []
    all_probs = []
    for i, level_name in enumerate(C.fpn_pyramid_levels):
        P_cls_logits = rpn_cls_outputs[i][0]
        P_cls = 1 / (1 + np.exp(-P_cls_logits)) # Terapkan sigmoid secara manual
        P_regr = rpn_regr_outputs[i][0]
        stride = C.fpn_strides[level_name]
        anchor_scales = C.anchor_box_scales[level_name]
        anchor_ratios = C.anchor_box_ratios

        (rows, cols) = P_cls.shape[:2]

        for anchor_ratio_idx, anchor_ratio in enumerate(anchor_ratios):
            for anchor_size_idx, anchor_size in enumerate(anchor_scales):
                anchor_idx = anchor_ratio_idx + len(anchor_ratios) * anchor_size_idx

                anchor_x, anchor_y = anchor_size * anchor_ratio[0], anchor_size * anchor_ratio[1]

                regr = P_regr[:, :, anchor_idx*4:anchor_idx*4+4]

                X_grid, Y_grid = np.meshgrid(np.arange(cols), np.arange(rows))

                cx = X_grid * stride + stride / 2
                cy = Y_grid * stride + stride / 2

                cx = cx + regr[:, :, 0] * anchor_x
                cy = cy + regr[:, :, 1] * anchor_y
                w = np.exp(regr[:, :, 2]) * anchor_x
                h = np.exp(regr[:, :, 3]) * anchor_y

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = x1 + w
                y2 = y1 + h

                proposals = np.stack([x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()], axis=1)
                all_proposals.append(proposals)
                all_probs.append(P_cls[:, :, anchor_idx].flatten())

    proposals = np.concatenate(all_proposals, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    proposals[:, 0] = np.clip(proposals[:, 0], 0, img_shape[1] - 1)
    proposals[:, 1] = np.clip(proposals[:, 1], 0, img_shape[0] - 1)
    proposals[:, 2] = np.clip(proposals[:, 2], 0, img_shape[1] - 1)
    proposals[:, 3] = np.clip(proposals[:, 3], 0, img_shape[0] - 1)

    valid_indices = np.where(probs > 0.7)[0]
    proposals = proposals[valid_indices, :]
    probs = probs[valid_indices]
    
    proposals, _ = non_max_suppression_fast(proposals, probs, overlap_thresh=0.7)

    return proposals

# --------------------------------------------------------------------
# BAGIAN 2: MEMUAT KONFIGURASI DAN MEMBANGUN MODEL (PENDEKATAN LOAD_MODEL)
# --------------------------------------------------------------------
print("Memuat file konfigurasi...")
try:
    with open(CONFIG_PATH, 'rb') as f_in:
        C = pickle.load(f_in)
except FileNotFoundError:
    raise FileNotFoundError(f"File konfigurasi '{CONFIG_PATH}' tidak ditemukan.")

CLASS_MAPPING = C.class_mapping
if CLASS_MAPPING is None:
    raise ValueError("File konfigurasi tidak memiliki 'class_mapping'.")
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
print("Konfigurasi dimuat. Mapping kelas:", IDX_TO_CLASS)

print(f"Memuat model lengkap dari '{MODEL_PATH}'...")
try:
    # Memuat model lengkap (arsitektur + bobot) dari file.
    # 'custom_objects' diperlukan agar Keras mengenali lapisan kustom kita.
    full_model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'RoiPoolingConv': RoiPoolingConv})
    print("Model lengkap berhasil dimuat.")
    # full_model.summary() # Uncomment baris ini jika perlu debugging nama lapisan
except Exception as e:
    print(f"\nError saat memuat model dari file H5. Pastikan file tidak korup.")
    raise Exception(f"Gagal memuat model dari '{MODEL_PATH}'. Error: {e}")

# --- Rekonstruksi model-model untuk inferensi dari model yang sudah dimuat ---
print("Mengekstrak dan membangun model RPN dan Classifier untuk inferensi...")

try:
    # --- Bangun model RPN ---
    image_input = full_model.input[0] # Input gambar
    
    # Cari nama lapisan output RPN dan P2 feature map dari model yang dimuat
    # Berdasarkan summary sebelumnya, P2 adalah input ke roi_pooling_conv
    roi_pooling_input_layer = full_model.get_layer('roi_pooling_conv').input[0].name.split('/')[0]
    p2_feature_map_output = full_model.get_layer(roi_pooling_input_layer).output

    # Output RPN adalah 8 lapisan terakhir sebelum 2 output classifier
    rpn_outputs = full_model.outputs[:-2]
    
    model_rpn_fpn = Model(inputs=image_input, outputs=rpn_outputs + [p2_feature_map_output])
    print("Model RPN berhasil dibangun.")

    # --- Bangun model Classifier ---
    # Kita perlu membuat ulang grafik classifier secara terpisah
    feature_map_input = Input(shape=p2_feature_map_output.shape[1:])
    roi_input_classifier = Input(shape=(C.num_rois, 4))
    
    # Gunakan kembali lapisan-lapisan dari model yang sudah dimuat
    roi_pooling_layer = full_model.get_layer('roi_pooling_conv')
    
    # Cari model classifier yang nested
    classifier_head_model = None
    for layer in full_model.layers:
        if layer.name.startswith('model'): # Biasanya bernama 'model', 'model_1', dst.
            classifier_head_model = layer
            break
    if classifier_head_model is None:
        raise ValueError("Tidak dapat menemukan model classifier head yang nested.")

    pooled_rois = roi_pooling_layer([feature_map_input, roi_input_classifier])
    classifier_outputs = classifier_head_model(pooled_rois)

    model_classifier_fpn = Model(inputs=[feature_map_input, roi_input_classifier], outputs=classifier_outputs)
    print("Model Classifier berhasil dibangun.")

except Exception as e:
    print("\nError saat membangun model inferensi dari model yang dimuat.")
    print("Cetak ringkasan model lengkap untuk debugging nama lapisan:")
    full_model.summary(line_length=120)
    raise e

print("Semua model siap untuk prediksi.")

# --------------------------------------------------------------------
# BAGIAN 3: FUNGSI PREPROCESSING DAN ROUTES FLASK
# --------------------------------------------------------------------
def format_img(img, C):
    """Mempersiapkan gambar untuk input model."""
    img_min_side = float(getattr(C, 'img_min_side', 600.0))
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        resized_height = int(ratio * height)
        resized_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        resized_width = int(ratio * width)
        resized_height = int(img_min_side)

    img_resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    img_resized = img_resized.astype(np.float32)
    img_resized[:, :, 0] -= C.img_channel_mean[0]
    img_resized[:, :, 1] -= C.img_channel_mean[1]
    img_resized[:, :, 2] -= C.img_channel_mean[2]
    img_resized /= C.img_scaling_factor
    return np.expand_dims(img_resized, axis=0), ratio

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    X, ratio = format_img(image_cv, C)

    rpn_and_p2_outputs = model_rpn_fpn.predict(X)

    num_fpn_levels = len(C.fpn_pyramid_levels)
    num_rpn_outputs_per_level = 2
    
    # Memisahkan output RPN dengan benar
    num_rpn_total_outputs = num_rpn_outputs_per_level * num_fpn_levels
    rpn_cls_outputs = [rpn_and_p2_outputs[i] for i in range(0, num_rpn_total_outputs, num_rpn_outputs_per_level)]
    rpn_regr_outputs = [rpn_and_p2_outputs[i] for i in range(1, num_rpn_total_outputs, num_rpn_outputs_per_level)]
    P2_feature_map = rpn_and_p2_outputs[-1]

    R = rpn_to_roi_fpn(rpn_cls_outputs, rpn_regr_outputs, C, X.shape[1:3])

    all_dets = []
    if R.shape[0] == 0:
        print("Tidak ada proposal yang dihasilkan oleh RPN.")
    else:
        R_padded = np.zeros((1, C.num_rois, 4))
        num_rois_to_process = min(R.shape[0], C.num_rois)
        
        R_xywh = R.copy()
        R_xywh[:, 2] -= R_xywh[:, 0] # width
        R_xywh[:, 3] -= R_xywh[:, 1] # height
        R_padded[0, :num_rois_to_process, :] = R_xywh[:num_rois_to_process, :]

        [P_cls, P_regr] = model_classifier_fpn.predict([P2_feature_map, R_padded])

        bboxes = {}
        probs = {}
        for ii in range(num_rois_to_process):
            if np.max(P_cls[0, ii, :]) < SCORE_THRESH or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = IDX_TO_CLASS[np.argmax(P_cls[0, ii, :])]
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = R_padded[0, ii, :]
            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass

            bboxes[cls_name].append([x, y, x + w, y + h])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for key in bboxes:
            bbox = np.array(bboxes[key])
            prob = np.array(probs[key])
            new_boxes, new_probs = non_max_suppression_fast(bbox, prob, overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                real_x1, real_y1, real_x2, real_y2 = (int(round(x1 / ratio)), int(round(y1 / ratio)), int(round(x2 / ratio)), int(round(y2 / ratio)))
                draw.rectangle([(real_x1, real_y1), (real_x2, real_y2)], outline="red", width=3)
                label = f"{key}: {new_probs[jk]:.2f}"
                draw.text((real_x1, real_y1 - 15), label, fill="red", font=font)
                all_dets.append({'class': key, 'prob': float(new_probs[jk])})

    result_path = os.path.join("static", "result.jpg")
    image.save(result_path)
    cache_buster = random.randint(0, 100000)

    return jsonify({
        'cars_detected': len(all_dets),
        'detections': all_dets,
        'image_url': f'/static/result.jpg?cb={cache_buster}'
    })

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

