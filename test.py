#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from model import style_extractor, extractor, num_content_layers, num_style_layers
from flask_cors import CORS, cross_origin
import base64
import cv2
from lib import *
import pickle
import pathlib
import urllib.request
# 모듈 구성 및 임포트
import IPython.display as display
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
from fixed import tensor_to_image, load_img, extractor, style_extractor, train_step, optimizer, style_content_loss
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/perform_nst": {"origins": "https://localhost:3000"}, r"/plant_disease": {"origins": "https://localhost:3000"}})

# 사용자 콘텐츠 업로드 및 NST 수행 엔드포인트
@app.route('/perform_nst', methods=['POST'])
@cross_origin(origins='https://localhost:3000', supports_credentials=True)
def perform_nst():
    # 사용자 콘텐츠 이미지 업로드
    content_image = request.files['content_image']
    # content_image = Image.open('./data/5.jpg')
    content_image = Image.open(content_image).convert("RGB")
    content_image = tf.keras.preprocessing.image.img_to_array(content_image)
    content_image = content_image / 255
    content_image = content_image[tf.newaxis, :]

    # 선택된 스타일 이미지 업로드
    style_image = request.files['style_image']
    # style_image = Image.open('./style/폴리곤_컬러.jpg')
    style_image = Image.open(style_image)
    style_image = tf.keras.preprocessing.image.img_to_array(style_image)
    style_image = style_image / 255
    style_image = style_image[tf.newaxis, :]

    # NST를 위한 설정 및 초기화
    style_outputs = style_extractor(style_image)
    results = extractor(content_image)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    new_image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    content_weight = 1
    style_weight = 1000
    num_epochs = 75
    verbose = 25

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            outputs = extractor(new_image)
            content_outputs = outputs['content']
            style_outputs = outputs['style']

            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
            content_loss *= content_weight / num_content_layers

            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
            style_loss *= style_weight / num_style_layers

            total_loss = content_loss + style_loss

        gradient = tape.gradient(total_loss, new_image)
        optimizer.apply_gradients([(gradient, new_image)])

        new_image.assign(tf.clip_by_value(new_image, 0.0, 1.0))

    # NST 결과 이미지를 PIL 이미지로 변환
    nst_result = tf.squeeze(new_image, axis=0)
    nst_result = tf.keras.preprocessing.image.array_to_img(nst_result)

    # PIL 이미지를 Base64 인코딩된 문자열로 변환
    img_byte_array = BytesIO()
    nst_result.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

    return jsonify({"imageData": img_base64})

@app.route('/plant_disease', methods=['POST'])
@cross_origin(origins='https://localhost:3000', supports_credentials=True)
def plant_disease():
    # 이미지 파일 불러오기
    img_path = request.json.get('imageUrl')
    print(img_path)

    resp = urllib.request.urlopen(img_path)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 모델 불러오기 (모델 파일의 경로를 입력해야 합니다)
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        model_path = 'C:\\Users\\smhrd\\Desktop\\HGY\\Flask\\Plant\\export.pkl'
        learn = load_learner(model_path)
    finally:
        pathlib.PosixPath = posix_backup

    # 예측 수행
    p = learn.predict(image)
    category = str(p[0]).replace("Category ", "")
    if category == "export.pkl":
        illname = "healthy"
    else:
        illname = category
    imageinfo = illname
    print(imageinfo)

    return imageinfo

@app.route('/styleTransfer', methods=['POST'])
@cross_origin(origins='https://localhost:3000', supports_credentials=True)
def styleTransfer():
    content_image_url = request.json['content_image']
    style_image_url = request.json['style_image']

    # 이미지 URL에서 이미지를 다운로드
    content_image_response = requests.get(content_image_url)
    style_image_response = requests.get(style_image_url)

    content_image = Image.open(BytesIO(content_image_response.content)).convert("RGB")
    style_image = Image.open(BytesIO(style_image_response.content)).convert("RGB")

    # 콘텐츠 이미지와 스타일 이미지를 NumPy 배열로 변환
    content_image = np.array(content_image)
    style_image = np.array(style_image)

    content_image = tf.convert_to_tensor(content_image, dtype=tf.float32)

    # 원래 이미지의 가로, 세로 크기
    original_height, original_width, _ = content_image.shape

    # 이미지 크기 조정 (원래 크기의 70%로 크기 조정)
    scale = 0.5
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    content_image = tf.image.resize(content_image, (new_height, new_width))
    content_image = content_image / 255
    content_image = content_image[tf.newaxis, :]

    # 스타일 이미지도 크기를 조정
    style_image = tf.convert_to_tensor(style_image, dtype=tf.float32)
    style_image = tf.image.resize(style_image, (new_height, new_width))
    style_image = style_image / 255
    style_image = style_image[tf.newaxis, :]

    # VGG19 모델을 불러오기
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    style_outputs = style_extractor(style_image * 255)

    results = extractor(content_image)

    # 스타일과 콘텐츠의 타깃값을 지정
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # 최적화시킬 이미지를 담을 tf.Variable을 정의하고 콘텐츠 이미지로 초기화
    # (이때 tf.Variable는 콘텐츠 이미지와 크기가 같아야 합니다.)
    image = tf.Variable(content_image)

    # 최적화를 진행
    import time
    start = time.time()

    epochs = 5
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = style_content_loss(outputs, style_targets, content_targets)
            grad = tape.gradient(loss, image)
            optimizer.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, 0.0, 1.0))

        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    # NST 결과 이미지를 PIL 이미지로 변환
    nst_result = tf.squeeze(image, axis=0)
    nst_result = tf.keras.preprocessing.image.array_to_img(nst_result)

    # PIL 이미지를 Base64 인코딩된 문자열로 변환
    img_byte_array = BytesIO()
    nst_result.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

    return jsonify({"imageData": img_base64})


if __name__ == '__main__':
     app.run(host="0.0.0.0", port="5000",  debug=False)
