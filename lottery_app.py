import os
import pandas as pd
import random
import requests
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from teachable_machine_eval import load_model, load_image_for_model
import threading
import time

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

    # 檢查 IP 是否被限制
    with lock:
        if client_ip in restricted_ips and time.time() < restricted_ips[client_ip]:
            remaining_time = int(restricted_ips[client_ip] - time.time())
            return jsonify({"status": "BLOCKED", "result": f"因先前擲出蓋杯，您的IP已被限制，請於 {remaining_time} 秒後再試"})
    
    possible_results = ["聖杯", "笑杯", "蓋杯"]

    # 讀取圖片並辨識
    img_array = load_image_for_model("latest.jpg")
    predictions = tm_model(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = possible_results[predicted_class]
    confidence_score = predictions[0][predicted_class]
    prediction_message = f"擲杯結果: {predicted_class_name} {np.round(confidence_score*100, 2)}"
    app.logger.info(prediction_message)

    # 如果是後續擲杯，計入 session
    if "current_poem" in session:
        session["throw_count"] = session.get("throw_count", 0) + 1
        if predicted_class_name == "聖杯":
            session["sacred_count"] = session.get("sacred_count", 0) + 1
            if session["sacred_count"] == 3:
                return jsonify({"status": "DONE", "result": "三次聖杯達成"})
            return jsonify({
                "status": "PENDING",
                "result": predicted_class_name,
                "sacred_count": session["sacred_count"]
            })
        else:
            session["throw_count"] = 0
            session["sacred_count"] = 0
            return jsonify({"status": "FAILED", "result": "非聖杯，需重新抽籤"})
    
    # 前置擲杯，直接返回結果
    if predicted_class_name == "蓋杯":
        with lock:
            restricted_ips[client_ip] = time.time() + 300  # 限制 5 分鐘
    return jsonify({"status": "PRE_THROW", "result": predicted_class_name})

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

@app.route("/")
def home():
    return render_template("pray_try.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)