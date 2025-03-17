import os
import time
import requests
from flask import jsonify

# 設定 ESP-32S 伺服器的 IP（請替換為實際的 ESP-32S IP）
ESP32_IP = "http://192.168.1.100"  # 請確認 ESP-32S 的內網 IP

# 設定擲茭影像儲存路徑
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 設定擲茭流程的狀態
PENDING = "PENDING"
PROCESSING = "PROCESSING"
DONE = "DONE"
current_status = PENDING
result = None  # 存放擲茭結果

def start_throw():
    """
    開始擲茭流程：
    1. 通知 ESP-32S 啟動視訊串流
    2. 等待 2 秒後，傳送 "THROW" 指令啟動擲茭機構
    3. ESP-32S 擷取影像，並回傳至 Flask 進行辨識
    4. Flask 使用 AI 模型進行辨識
    """
    global current_status, result
    current_status = PROCESSING
    result = None  # 重置擲茭結果

    # 1️⃣ 啟動 ESP-32S 串流
    try:
        requests.get(f"{ESP32_IP}/start_stream")
        print("[INFO] ESP-32S 影像串流啟動")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 無法啟動 ESP-32S 串流: {e}")
        current_status = PENDING
        return jsonify({"status": "error", "message": "ESP-32S 無法啟動串流"}), 500

    # 2️⃣ 等待 2 秒，確保視訊連線穩定
    time.sleep(2)

    # 3️⃣ 發送擲茭指令至 ESP-32S
    try:
        requests.get(f"{ESP32_IP}/throw")
        print("[INFO] 擲茭指令已發送")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 無法發送擲茭指令: {e}")
        current_status = PENDING
        return jsonify({"status": "error", "message": "ESP-32S 擲茭指令發送失敗"}), 500

    return jsonify({"status": "processing", "message": "擲茭中"}), 200

def check_result():
    """
    查詢擲茭結果：
    1. 若結果仍在處理，回傳 "PENDING"
    2. 若辨識完成，回傳 "DONE" 並附帶結果（A/B/C）
    """
    global current_status, result

    if current_status == PENDING:
        return jsonify({"status": PENDING}), 200

    if current_status == PROCESSING:
        return jsonify({"status": PROCESSING}), 200

    if current_status == DONE:
        return jsonify({"status": DONE, "result": result}), 200

    return jsonify({"status": "error", "message": "未知錯誤"}), 500

def set_result(predicted_result):
    """
    設定擲茭結果：
    1. 由 AI 辨識模組呼叫，將結果存入變數
    2. 通知 ESP-32S 可關閉串流，並顯示擲茭結果
    """
    global current_status, result
    current_status = DONE
    result = predicted_result

    # 通知 ESP-32S 顯示結果
    try:
        requests.get(f"{ESP32_IP}/show_result?result={result}")
        print(f"[INFO] 擲茭結果已傳送至 ESP-32S: {result}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 無法傳送擲茭結果至 ESP-32S: {e}")

    return jsonify({"status": "success", "message": f"擲茭結果設定為 {result}"}), 200
