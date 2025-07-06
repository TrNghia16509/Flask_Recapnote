# flask_api.py
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def process_audio_backend(filepath):
    model = WhisperModel("small", compute_type="int8")
    segments, info = model.transcribe(filepath, language="vi")
    full_text = "\n".join([seg.text for seg in segments])
    subject = genai.GenerativeModel("gemini-3.5-flash").generate_content(
        "Chủ đề chính là gì?\n" + full_text).text.strip()
    summary = genai.GenerativeModel("gemini-3.5-flash").generate_content(
        "Tóm tắt:\n" + full_text).text.strip()
    return subject, summary, full_text

@app.route("https://flask-recapnote.onrender.com/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        subject, summary, full_text = process_audio_backend(tmp.name)
        return jsonify({
            "subject": subject,
            "summary": summary,
            "text": full_text
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
