import os
import pandas as pd
import random
import requests
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, session, Response
from modules.teachable_machine_eval import load_model, load_image_for_model
import threading
import cv2
import logging
import subprocess


app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app.secret_key = "your_secret_key"
if not OPENAI_API_KEY:
    raise ValueError("請設定環境變數 OPENAI_API_KEY")

lottery_data = pd.read_csv("lottery.csv")
restricted_ips = {}
lock = threading.Lock()
tm_model = load_model("model.savedmodel")

@app.route("/draw_lottery", methods=["POST"])
def draw_lottery():
    print("[DEBUG] 抽籤請求收到")
    data = request.json or {}
    exclude_ids = data.get("exclude_ids", [])

    # 過濾掉已抽到的籤詩
    available_lottery_data = lottery_data[~lottery_data["id"].isin(exclude_ids)]
    if available_lottery_data.empty:
        return jsonify({"status": "error", "message": "無可抽取的籤詩"}), 400

    selected = available_lottery_data.sample().iloc[0]
    poem_data = {
        "id": int(selected["id"]),
        "poem": selected["poem"],
        "fate": selected["fate"],
        "marriage": selected["marriage"],
        "interpretation": selected["interpretation"]
    }
    session["current_poem"] = poem_data
    session["throw_count"] = 0
    session["sacred_count"] = 0
    return jsonify({
        "status": "success",
        "id": poem_data["id"],
        "poem": poem_data["poem"]
    })

@app.route("/start_throw", methods=["POST"])
def start_throw():
    print("[DEBUG] 擲杯請求收到")
    client_ip = request.remote_addr

    with lock:
        if client_ip in restricted_ips and time.time() < restricted_ips[client_ip]:
            remaining_time = int(restricted_ips[client_ip] - time.time())
            return jsonify({"status": "BLOCKED", "result": f"因先前擲出蓋杯，您的IP已被限制，請於 {remaining_time} 秒後再試"})

    possible_results = ["聖杯", "笑杯", "蓋杯"]
    #image_files = ["positive.jpg", "negative.jpg", "undefined.jpg"] # 完整杯型
    #image_files = ["positive.jpg", "undefined.jpg"] # 沒有蓋杯
    #image_files = ["positive.jpg"] 只有聖杯
    #selected_image = random.choice(image_files)

    #app.logger.info(f"選中的圖片: {selected_image}")

    try:
        img_array = load_image_for_model("latest.jpg")
        #img_array = load_image_for_model(selected_image)
    except Exception as e:
        app.logger.error(f"圖片載入失敗: {str(e)}")
        return jsonify({"status": "error", "result": f"擲杯圖片載入失敗（{img_array}），請確認圖片是否存在"}), 500
        #return jsonify({"status": "error", "result": f"擲杯圖片載入失敗（{selected_image}），請確認圖片是否存在"}), 500

    predictions = tm_model(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = possible_results[predicted_class]
    confidence_score = predictions[0][predicted_class]
    prediction_message = f"擲杯結果: {predicted_class_name} {np.round(confidence_score*100, 2)}"
    app.logger.info(prediction_message)

    data = request.json or {}
    is_pre_throw = data.get("is_pre_throw", False)
    is_can_ask = data.get("is_can_ask", False)  # 新增標誌

    if not is_pre_throw and "current_poem" in session:
        # 後續擲杯邏輯
        session["throw_count"] = session.get("throw_count", 0) + 1
        if predicted_class_name == "聖杯":
            session["sacred_count"] = session.get("sacred_count", 0) + 1
            if session["sacred_count"] == 3:
                return jsonify({"status": "DONE", "result": "三次聖杯達成，請擲杯，看神明是否指示繼續抽籤"})
            return jsonify({
                "status": "PENDING",
                "result": predicted_class_name,
                "sacred_count": session["sacred_count"]
            })
        else:
            session["throw_count"] = 0
            session["sacred_count"] = 0
            return jsonify({"status": "FAILED", "result": "非聖杯，需重新抽籤"})
    else:
        # 前置擲杯邏輯
        if predicted_class_name == "蓋杯" and not is_can_ask:  # 只有非 canAskBtn 的蓋杯才限制 IP
            with lock:
                restricted_ips[client_ip] = time.time() + 10  # 限制 5 分鐘
        result = {"status": "PRE_THROW", "result": predicted_class_name}
        print("[DEBUG] 擲杯回傳:", result)
        return jsonify(result)

@app.route("/get_current_poem", methods=["GET"])
def get_current_poem():
    poem_data = session.get("current_poem", {})
    return jsonify({
        "id": poem_data["id"],
        "poem": poem_data["poem"]
    })

@app.route("/interpret_lottery", methods=["POST"])
def interpret_lottery():
    print("[DEBUG] 解籤請求收到")
    data = request.json
    user_name = data.get("name", "")
    user_birth = data.get("birth", "未提供")
    user_address = data.get("address", "")
    user_question = data.get("question", "請解釋這首籤詩")
    poem_ids = data.get("poem_ids", [])

    # 從 lottery_data 中獲取所有籤詩資料
    poems_data = []
    for poem_id in poem_ids:
        poem_row = lottery_data[lottery_data["id"] == poem_id].iloc[0]
        poem_data = {
            "id": int(poem_row["id"]),
            "poem": poem_row["poem"],
            "fate": poem_row["fate"],
            "marriage": poem_row["marriage"],
            "interpretation": poem_row["interpretation"]
        }
        poems_data.append(poem_data)

    # 組合所有籤詩資訊，供綜合解釋
    poem_sequence = ", ".join([f"籤詩 ID {poem['id']}：{poem['poem']}" for poem in poems_data])
    prompt = f"""
    你是一位親切且知識淵博的解籤師，請根據以下籤詩內容，以自然的人類口吻回答使用者的問題。
    使用者姓名：{user_name}
    使用者生辰：{user_birth}
    使用者地址：{user_address}
    使用者依序抽到的籤為：{poem_sequence}
    緣份：{[poem['fate'] for poem in poems_data]}
    婚姻：{[poem['marriage'] for poem in poems_data]}
    傳統解釋：{[poem['interpretation'] for poem in poems_data]}
    使用者問題：{user_question}
    請綜合所有籤詩，給出一個整體的解釋。
    """

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一位解籤師，擅長用親切語氣解釋籤詩。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,  # 增加 token 數以容納綜合解釋
        "temperature": 0.7
    }

    try:
        time.sleep(1)
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 429:
            interpretation = "解籤失敗：請求過於頻繁，請稍後再試。"
        else:
            response.raise_for_status()
            interpretation = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        interpretation = f"解籤失敗：{str(e)}"

    session["throw_count"] = 0
    session["sacred_count"] = 0
    return jsonify({
        "poems": poems_data,
        "interpretation": interpretation
    })

@app.route("/check_ip_restriction", methods=["GET"])
def check_ip_restriction():
    client_ip = request.remote_addr
    with lock:
        if client_ip in restricted_ips and time.time() < restricted_ips[client_ip]:
            remaining_time = int(restricted_ips[client_ip] - time.time())
            return jsonify({"status": "BLOCKED", "remaining_time": remaining_time})
        return jsonify({"status": "OK"})
    
# 確保同層資料夾存在
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    app.logger.info("收到上傳圖片請求")
    if 'image' not in request.files:
        app.logger.error("未找到圖片")
        return jsonify({"status": "error", "message": "未找到圖片"}), 400
    
    file = request.files['image']
    if file.filename == '':
        app.logger.error("檔案名稱為空")
        return jsonify({"status": "error", "message": "檔案名稱為空"}), 400
    
    # 儲存為 latest.jpg
    file_path = os.path.join(UPLOAD_FOLDER, 'latest.jpg')
    file.save(file_path)
    app.logger.info(f"圖片已儲存到 {file_path}")
    
    return jsonify({"status": "success", "message": "圖片已儲存"}), 200

@app.route("/")
def home():
    return render_template("divination.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)