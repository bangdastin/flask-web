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

# Impor yang diperlukan dari keras_frcnn
from keras_frcnn import roi_helpers
from keras_frcnn.config import Config

# -------------------------------------------------------
# KONFIGURASI
# -------------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

# Pastikan backend Keras menggunakan format data TensorFlow (channels_last)
K.set_image_data_format('channels_last')

# --- PERBAIKAN: Pengelolaan Sesi dan Grafik TensorFlow ---
# Inisialisasi sesi TensorFlow secara global
# Ini memastikan semua operasi Keras terjadi dalam sesi dan grafik yang sama
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)
K.set_session(sess)
# Simpan grafik default untuk digunakan nanti di dalam request
graph = tf.compat.v1.get_default_graph()
# ---------------------------------------------------------

MODEL_PATH = "model_frcnn_vgg.hdf5"
CONFIG_PATH = "model_vgg_config.pickle"
SCORE_THRESH = 0.7

# -------------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------------
try:
    with open(CONFIG_PATH, 'rb') as f_in:
        C = pickle.load(f_in)
except FileNotFoundError:
    raise FileNotFoundError(f"File konfigurasi tidak ditemukan di {CONFIG_PATH}.")

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

CLASS_MAPPING = C.class_mapping
if 'bg' not in CLASS_MAPPING:
    CLASS_MAPPING['bg'] = len(CLASS_MAPPING)
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
print("Kelas yang terdeteksi oleh model:", IDX_TO_CLASS)

# -------------------------------------------------------
# BUILD MODEL
# -------------------------------------------------------
# Pastikan model dibuat di dalam grafik yang sama
with graph.as_default():
    K.set_session(sess)
    
    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    
    from keras_frcnn import vgg as nn

    shared_layers = nn.nn_base(img_input, trainable=True)
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    feature_map_input = Input(shape=(None, None, 512))
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois,
                               nb_classes=len(CLASS_MAPPING), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    print(f"Memuat bobot dari {MODEL_PATH}...")
    try:
        model_rpn.load_weights(MODEL_PATH, by_name=True)
        model_classifier_only.load_weights(MODEL_PATH, by_name=True)
    except Exception as e:
        print(f"Error saat memuat bobot: {e}")

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier_only.compile(optimizer='sgd', loss='mse')

# -------------------------------------------------------
# FUNGSI BANTUAN
# -------------------------------------------------------
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
    img_resized = img_resized.astype(np.float32)
    img_resized[:, :, 0] -= C.img_channel_mean[0]
    img_resized[:, :, 1] -= C.img_channel_mean[1]
    img_resized[:, :, 2] -= C.img_channel_mean[2]
    img_resized /= C.img_scaling_factor
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized, ratio

# -------------------------------------------------------
# ROUTES FLASK
# -------------------------------------------------------
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

    # --- PERBAIKAN: Gunakan grafik dan sesi yang sudah disimpan ---
    with graph.as_default():
        K.set_session(sess)
        
        [Y1, Y2, F] = model_rpn.predict(X)

        # --- PERBAIKAN: Sesuaikan pemanggilan rpn_to_roi dengan versi library Anda ---
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
        
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < SCORE_THRESH or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = IDX_TO_CLASS[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]
                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                
                bboxes[cls_name].append([C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            
            real_x1 = int(round(x1 / ratio))
            real_y1 = int(round(y1 / ratio))
            real_x2 = int(round(x2 / ratio))
            real_y2 = int(round(y2 / ratio))

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
    app.run(host='0.0.0.0', port=5000, debug=True)
