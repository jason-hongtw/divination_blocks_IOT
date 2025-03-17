import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# 初始化 Flask 伺服器
app = Flask(__name__)

# 設定圖片上傳資料夾
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 載入訓練好的擲茭辨識模型
MODEL_PATH = "thrown_cup_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 定義擲茭結果類別名稱（與模型輸出對應）
class_names = ["A_sacred", "B_laughing", "C_angry"]

def predict_cup(image_path):
    """
    讀取影像並使用 CNN 模型進行擲茭結果辨識
    :param image_path: 上傳影像的檔案路徑
    :return: 擲茭結果類別（A_sacred, B_laughing, C_angry）
    """
    img = cv2.imread(image_path)  # 讀取圖片
    img = cv2.resize(img, (224, 224)) / 255.0  # 調整大小 & 標準化
    img = np.expand_dims(img, axis=0)  # 增加維度，符合模型輸入格式

    preds = model.predict(img)  # 進行預測
    predicted_class = np.argmax(preds, axis=1)[0]  # 取得最高機率的分類索引
    
    return class_names[predicted_class]  # 回傳預測結果

# API 1: 上傳 ESP-32S 擷取的擲茭影像
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """
    ESP-32S 透過此 API 上傳擲茭影像
    影像將被存入 uploads/latest.jpg，供後續辨識
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "未收到影像"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    file.save(file_path)  # 儲存上傳的圖片

    return jsonify({"status": "success", "message": "圖片上傳成功"}), 200

# API 2: 選擇祈福
@app.route("/pray")
def pray_select():
    return render_template("pray_select.html")

@app.route("/pray/<category>")
def pray_page(category):
    if category == "love":  # 目前開放「姻緣」
        return render_template("pray.html", category="姻緣")
    else:
        return "尚未開放", 404

# API 3: 顯示擲茭網頁（可視化介面）
@app.route("/")
def home():
    """
    提供擲茭的主網頁
    信徒可以在此觀看擲茭結果（未來擴充）
    """
    return render_template("index.html")

# 啟動 Flask 伺服器
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
