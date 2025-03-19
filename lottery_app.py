import os
import pandas as pd
import random
import requests
import time
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = "your_secret_key"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("請設定環境變數 OPENAI_API_KEY")

lottery_data = pd.read_csv("lottery.csv")

@app.route("/draw_lottery", methods=["POST"])
def draw_lottery():
    print("[DEBUG] 抽籤請求收到")
    selected = lottery_data.sample().iloc[0]
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
    return jsonify({"status": "success", "poem": poem_data["poem"]})

@app.route("/start_throw", methods=["POST"])
def start_throw():
    print("[DEBUG] 擲茭請求收到")
    possible_results = ["聖茭", "B_laughing", "C_angry"]
    result = random.choice(possible_results)
    
    session["throw_count"] = session.get("throw_count", 0) + 1
    if result == "聖茭":
        session["sacred_count"] = session.get("sacred_count", 0) + 1
        if session["sacred_count"] == 3:
            return jsonify({"status": "DONE", "result": "三次聖茭達成"})
        return jsonify({
            "status": "PENDING",
            "result": result,
            "sacred_count": session["sacred_count"]
        })
    else:
        session["throw_count"] = 0
        session["sacred_count"] = 0
        return jsonify({"status": "FAILED", "result": "非聖茭，需重新抽籤"})

@app.route("/interpret_lottery", methods=["POST"])
def interpret_lottery():
    print("[DEBUG] 解籤請求收到")
    user_question = request.json.get("question", "請解釋這首籤詩")
    poem_data = session.get("current_poem", {})
    
    prompt = f"""
    你是一位親切且知識淵博的解籤師，請根據以下籤詩內容，以自然的人類口吻回答使用者的問題。
    籤詩：{poem_data.get("poem", "")}
    緣份：{poem_data.get("fate", "")}
    婚姻：{poem_data.get("marriage", "")}
    傳統解釋：{poem_data.get("interpretation", "")}
    使用者問題：{user_question}
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
        "max_tokens": 300,
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
        "id": poem_data["id"],
        "poem": poem_data["poem"],
        "interpretation": interpretation
    })

@app.route("/")
def home():
    return render_template("pray_try.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)