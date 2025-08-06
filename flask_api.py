from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route("/")
def home():
    return "✅ Flask backend is running."

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)

        # Process bằng Whisper
        model = WhisperModel("small", compute_type="int8")
        segments, info = model.transcribe(tmp.name, language="vi")
        full_text = "\n".join([seg.text for seg in segments])

        # Gemini AI xử lý chủ đề và tóm tắt
        subject = genai.GenerativeModel("gemini-3.5-flash").generate_content(
            "Chủ đề chính là gì?\n" + full_text).text.strip()

        summary = genai.GenerativeModel("gemini-3.5-flash").generate_content(
            "Tóm tắt:\n" + full_text).text.strip()

        return jsonify({
            "subject": subject,
            "summary": summary,
            "text": full_text
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

