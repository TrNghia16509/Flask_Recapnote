from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS
from faster_whisper import WhisperModel
import fitz, docx, tempfile, os
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ============= Config ============
app = Flask(__name__)
CORS(app)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
whisper_model = WhisperModel("small", compute_type="int8")  # Load 1 lần

@app.route("/")
def home():
    return "✅ Flask backend is running."

# === Backblaze setup ===
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account(
    "production",
    os.getenv("B2_APPLICATION_KEY_ID"),
    os.getenv("B2_APPLICATION_KEY")
)
bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

# === Helpers ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def upload_to_b2(local_path, b2_filename, content_type):
    with open(local_path, "rb") as f:
        bucket.upload_bytes(f.read(), b2_filename, content_type=content_type)
    return f"{os.getenv('B2_PUBLIC_URL')}/{b2_filename}"

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
            segments, info_w = whisper_model.transcribe(tmp.name, language="vi")
            text = "\n".join([seg.text for seg in segments])
            content_type = "audio/wav"
        elif ext.endswith(".pdf"):
            text = extract_text_from_pdf(tmp.name)
            content_type = "application/pdf"
        elif ext.endswith(".docx"):
            text = extract_text_from_docx(tmp.name)
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            os.remove(tmp.name)
            return jsonify({"error": "Định dạng không hỗ trợ"}), 400

    # AI xử lý
    model = genai.GenerativeModel("gemini-1.5-flash")
    subject = model.generate_content(
        f"Chủ đề chính của nội dung sau là gì? {text}"
    ).text.strip()
    summary = model.generate_content(
        f"Bạn là chuyên gia về {subject}. Tóm tắt nội dung: {text}"
    ).text.strip()

    # Lưu file gốc lên B2
    b2_file_name = f"uploads/{file.filename}"
    file_url = upload_to_b2(tmp.name, b2_file_name, content_type)

    # Lưu JSON kết quả lên B2
    result_data = {
        "subject": subject,
        "summary": summary,
        "full_text": text,
        "file_url": file_url
    }
    json_name = f"results/{os.path.splitext(file.filename)[0]}.json"
    bucket.upload_bytes(json.dumps(result_data).encode("utf-8"), json_name, content_type="application/json")
    json_url = f"{os.getenv('B2_PUBLIC_URL')}/{json_name}"

    os.remove(tmp.name)

    return jsonify({
        "subject": subject,
        "summary": summary,
        "full_text": text,
        "file_url": file_url,
        "json_url": json_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
