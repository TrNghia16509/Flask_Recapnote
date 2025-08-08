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
whisper_model = WhisperModel("small", compute_type="int8")  # Load 1 lần

@app.route("/")
def home():
    return "✅ Flask backend is running."

# === Helper ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# === API ===
@app.route("/process_file", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return jsonify({"error": "Thiếu file"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        ext = file.filename.lower()

        if ext.endswith((".mp3", ".wav")):
            segments, info = whisper_model.transcribe(tmp.name, language="vi")
            text = "\n".join([seg.text for seg in segments])
        elif ext.endswith(".pdf"):
            text = extract_text_from_pdf(tmp.name)
        elif ext.endswith(".docx"):
            text = extract_text_from_docx(tmp.name)
        else:
            os.remove(tmp.name)
            return jsonify({"error": "Định dạng không hỗ trợ"}), 400

    os.remove(tmp.name)

    # === Gemini xử lý ===
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        subject = model.generate_content(
            f"Chủ đề chính của nội dung sau là gì? {text}"
        ).text.strip()

        summary = model.generate_content(
            f"Bạn là chuyên gia về {subject}. Tóm tắt nội dung: {text}"
        ).text.strip()
    except Exception as e:
        return jsonify({"error": f"Lỗi AI: {str(e)}"}), 500

    return jsonify({
        "subject": subject,
        "summary": summary,
        "full_text": text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


