from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

ESP32_IP = "http://192.168.1.167/"  # 替換成你的 ESP32 IP

@app.route("/")
def index():
    return render_template("arduino_try.html", esp32_ip=ESP32_IP)

@app.route("/command", methods=["POST"])
def send_command():
    command = request.form.get("command")
    if command not in ["THROW", "CLOSE"]:
        return jsonify({"status": "error", "message": "無效指令"}), 400

    try:
        esp_response = requests.get(f"{ESP32_IP}/command?cmd={command}", timeout=5)
        return jsonify({"status": "success", "esp_response": esp_response.text})
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "message": f"ESP32 無回應: {e}"}), 500

if __name__ == "main":
    app.run(host="0.0.0.0", port=5000, debug=True)