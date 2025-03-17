import os
import cv2
import numpy as np
import tensorflow as tf
from flask import jsonify
from modules.prayers import set_result  # 導入 prayers.py 設定擲茭結果的函式

# 設定擲茭影像資料夾
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "thrown_cup_classifier.h5"

# 確保上傳資料夾存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 載入預訓練好的擲茭辨識模型
model = tf.keras.models.load_model(MODEL_PATH)

# 定義擲茭類別名稱（與模型輸出對應）
class_names = ["A_sacred", "B_laughing", "C_angry"]

def predict_cup(image_path):
    """
    讀取影像並使用 CNN 模型進行擲茭結果辨識
    :param image_path: 上傳影像的檔案路徑
    :return: 擲茭結果類別（A_sacred, B_laughing, C_angry）
    """
    if not os.path.exists(image_path):
        return None  # 如果圖片不存在，返回 None

    img = cv2.imread(image_path)  # 讀取圖片
    img = cv2.resize(img, (224, 224)) / 255.0  # 調整大小 & 標準化
    img = np.expand_dims(img, axis=0)  # 增加維度，符合模型輸入格式

    preds = model.predict(img)  # 進行預測
    predicted_class = np.argmax(preds, axis=1)[0]  # 取得最高機率的分類索引

    return class_names[predicted_class]  # 回傳預測結果

def process_uploaded_frame():
    """
    處理 ESP-32S 上傳的影像，進行擲茭辨識並更新結果
    """
    image_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")

    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "沒有上傳的影像"}), 400

    # 執行擲茭辨識
    predicted_result = predict_cup(image_path)

    if predicted_result is None:
        return jsonify({"status": "error", "message": "影像辨識失敗"}), 500

    # 更新擲茭結果
    set_result(predicted_result)

    return jsonify({"status": "success", "result": predicted_result}), 200
