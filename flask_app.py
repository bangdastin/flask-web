# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import cv2
import pickle

import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model

from keras_frcnn import config, roi_helpers, vgg as nn
from keras_frcnn.config import Config  # <-- Tambahan penting

# -------------------------------------------------------
# KONFIGURASI
# -------------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_PATH = "model_frcnn_vgg.hdf5"
CONFIG_PATH = "model_vgg_config.pickle"
SCORE_THRESH = 0.5
NMS_THRESH = 0.3

# TF session (TF 1.14)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)
K.set_session(sess)
graph = tf.compat.v1.get_default_graph()

# -------------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------------
with open(CONFIG_PATH, 'rb') as f:
    C = pickle.load(f)

# Non-augmentasi saat inferensi
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

CLASS_MAPPING = C.class_mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
BG_CLASS_IDX = CLASS_MAPPING['bg']

# Target kelas: mobil atau car
if 'mobil' in CLASS_MAPPING:
    TARGET_CLASSES = {'mobil'}
elif 'car' in CLASS_MAPPING:
    TARGET_CLASSES = {'car'}
else:
    TARGET_CLASSES = set([k for k in CLASS_MAPPING.keys() if k != 'bg'])

# -------------------------------------------------------
# BUILD MODEL
# -------------------------------------------------------
img_input = Input(shape=(None, None, 3))
roi_input = Input(shape=(None, 4))
shared_layers = nn.nn_base(img_input, trainable=True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(shared_layers, roi_input, C.num_rois,
                           nb_classes=len(CLASS_MAPPING), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

model_rpn.load_weights(MODEL_PATH, by_name=True)
model_classifier.load_weights(MODEL_PATH, by_name=True)

# -------------------------------------------------------
# FUNGSI
# -------------------------------------------------------
def get_real_coordinates(ratio, x1, y1, x2, y2):
    return (int(round(x1 / ratio)),
            int(round(y1 / ratio)),
            int(round(x2 / ratio)),
            int(round(y2 / ratio)))

def format_img(img, C):
    img_min_side = float(C.im_size)
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
    img_resized = img_resized[:, :, ::-1].astype(np.float32)
    img_resized[:, :, 0] -= C.img_channel_mean[0]
    img_resized[:, :, 1] -= C.img_channel_mean[1]
    img_resized[:, :, 2] -= C.img_channel_mean[2]
    img_resized /= C.img_scaling_factor

    return np.expand_dims(img_resized, axis=0), ratio

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    X, ratio = format_img(image_np, C)

    with graph.as_default():
        K.set_session(sess)
        Y1, Y2 = model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, use_regr=True, overlap_thresh=0.7, dim_ordering='tf')

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes, probs = {}, {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                continue
            if ROIs.shape[1] < C.num_rois:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            P_cls, P_regr = model_classifier.predict([X, ROIs])

            for ii in range(P_cls.shape[1]):
                max_prob = np.max(P_cls[0, ii, :])
                cls_idx = np.argmax(P_cls[0, ii, :])

                if max_prob < SCORE_THRESH or cls_idx == BG_CLASS_IDX:
                    continue

                cls_name = IDX_TO_CLASS[cls_idx]
                if cls_name not in TARGET_CLASSES:
                    continue

                (x, y, w, h) = ROIs[0, ii, :]
                try:
                    tx = P_regr[0, ii, 4 * cls_idx]
                    ty = P_regr[0, ii, 4 * cls_idx + 1]
                    tw = P_regr[0, ii, 4 * cls_idx + 2]
                    th = P_regr[0, ii, 4 * cls_idx + 3]

                    cx, cy = x + w / 2.0, y + h / 2.0
                    cx1 = tx * w + cx
                    cy1 = ty * h + cy
                    w1 = np.exp(tw) * w
                    h1 = np.exp(th) * h

                    x1 = cx1 - w1 / 2.0
                    y1 = cy1 - h1 / 2.0
                    x2 = cx1 + w1 / 2.0
                    y2 = cy1 + h1 / 2.0

                    if np.any(np.isnan([x1, y1, x2, y2])) or w1 <= 0 or h1 <= 0:
                        continue
                except:
                    continue

                x1, y1, x2, y2 = get_real_coordinates(ratio, x1, y1, x2, y2)
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                bboxes[cls_name].append([x1, y1, x2, y2])
                probs[cls_name].append(max_prob)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = None

    cars_detected = 0
    for cls_name in bboxes:
        boxes = np.array(bboxes[cls_name])
        ps = np.array(probs[cls_name])
        nms_boxes, nms_probs = roi_helpers.non_max_suppression_fast(
            boxes, ps, overlap_thresh=NMS_THRESH)

        for (x1, y1, x2, y2), p in zip(nms_boxes, nms_probs):
            if cls_name in TARGET_CLASSES:
                cars_detected += 1
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            label = f"{cls_name} {p:.2f}"
            draw.text((x1, max(0, y1 - 12)), label, fill="red", font=font)

    result_path = os.path.join("static", "result.jpg")
    image.save(result_path)

    cache_buster = random.randint(0, 100000)
    return jsonify({
        'cars_detected': cars_detected,
        'image_url': f'/static/result.jpg?cb={cache_buster}'
    })

if __name__ == '__main__':
    app.run(debug=True)
