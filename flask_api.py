from flask import Flask, request, jsonify
import os, tempfile, json, docx, requests, urllib.parse
from dotenv import load_dotenv
from flask_cors import CORS
import pdfplumber
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from groq import Groq

# ============= Config ============
app = Flask(__name__)
CORS(app)
load_dotenv()

# Kiểm tra biến môi trường bắt buộc
required_env = ["GROQ_API_KEY", "ASSEMBLYAI_API_KEY", "B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME"]
for env_var in required_env:
    if not os.getenv(env_var):
        raise RuntimeError(f"❌ Missing environment variable: {env_var}")

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# AssemblyAI
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIBE_URL = "https://api.assemblyai.com/v2/transcript"

# Backblaze B2 (Private Bucket)
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", os.getenv("B2_APPLICATION_KEY_ID"), os.getenv("B2_APPLICATION_KEY"))
bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

# === Helpers ===
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def upload_to_b2(local_path, b2_filename, content_type):
    with open(local_path, "rb") as f:
        bucket.upload_bytes(f.read(), b2_filename, content_type=content_type)
    return b2_filename

def get_signed_url(file_name, valid_seconds=3600):
    """Tạo signed URL tạm thời cho file private"""
    auth_token = bucket.get_download_authorization(
        file_name_prefix=file_name,
        valid_duration_in_seconds=valid_seconds
    )
    base_url = b2_api.account_info.get_download_url()
    download_url = f"{base_url}/file/{bucket.name}/{urllib.parse.quote(file_name)}"
    return f"{download_url}?Authorization={auth_token}"

def transcribe_with_assemblyai(file_path, language_code):
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    with open(file_path, "rb") as f:
        upload_res = requests.post(ASSEMBLYAI_UPLOAD_URL, headers=headers, data=f)
    upload_res.raise_for_status()
    audio_url = upload_res.json()["upload_url"]

    payload = {
        "audio_url": audio_url,
        "language_code": None if language_code == "auto" else language_code
    }
    trans_res = requests.post(ASSEMBLYAI_TRANSCRIBE_URL, headers=headers, json=payload)
    trans_res.raise_for_status()
    transcript_id = trans_res.json()["id"]

    while True:
        status_res = requests.get(f"{ASSEMBLYAI_TRANSCRIBE_URL}/{transcript_id}", headers=headers)
        status_res.raise_for_status()
        status_data = status_res.json()
        if status_data["status"] == "completed":
            return status_data["text"]
        elif status_data["status"] == "error":
            raise Exception(f"AssemblyAI Error: {status_data['error']}")

def groq_generate(prompt, max_tokens=1000):
    """Gọi Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return {"status": "✅ Flask backend is running"}, 200

@app.route("/process_file", methods=["POST"])
def process_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Thiếu file"}), 400

        file = request.files["file"]
        language_code = request.form.get("language_code", "auto")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            ext = file.filename.lower()

            if ext.endswith((".mp3", ".wav")):
                text = transcribe_with_assemblyai(tmp.name, language_code)
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

        # AI xử lý với Groq
        subject = groq_generate(f"Hãy cho biết chủ đề chính của nội dung sau: {text}")
        summary = groq_generate(f"Tóm tắt nội dung sau một cách ngắn gọn, đủ ý trong 1000 từ:\n\n{text}")

        # Upload file gốc
        safe_file_name = f"uploads/{file.filename}"
        upload_to_b2(tmp.name, safe_file_name, content_type)
        file_url = get_signed_url(safe_file_name)

        # Upload JSON kết quả
        result_data = {
            "subject": subject,
            "summary": summary,
            "full_text": text,
            "file_url": file_url
        }
        json_name = f"results/{os.path.splitext(file.filename)[0]}.json"
        bucket.upload_bytes(json.dumps(result_data, ensure_ascii=False).encode("utf-8"),
                            json_name, content_type="application/json")
        json_url = get_signed_url(json_name)

        os.remove(tmp.name)
        return jsonify({
            "subject": subject,
            "summary": summary,
            "full_text": text,
            "file_url": file_url,
            "json_url": json_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Lấy signed URL mới cho file bất kỳ
@app.route("/get_signed_url", methods=["GET"])
def api_get_signed_url():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "Thiếu file_name"}), 400
    try:
        return jsonify({"signed_url": get_signed_url(file_name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Lấy nội dung JSON từ bucket private
@app.route("/get_json_content", methods=["GET"])
def get_json_content():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "Thiếu file_name"}), 400
    try:
        signed_url = get_signed_url(file_name)
        res = requests.get(signed_url)
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

