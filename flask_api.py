from flask import Flask, request, jsonify
import os, tempfile, json, docx, requests, urllib.parse
from dotenv import load_dotenv
from flask_cors import CORS
import pdfplumber
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from groq import Groq
import time 
from pydub import AudioSegment
import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "vinai/PhoWhisper-medium"   # có thể thay bằng PhoWhisper-small/large

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1,
)

# ============= Config ============
app = Flask(__name__)
CORS(app)
load_dotenv()

# Kiểm tra biến môi trường bắt buộc
required_env = ["ASSEMBLYAI_API_KEY", "B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME"]
for env_var in required_env:
    if not os.getenv(env_var):
        raise RuntimeError(f"❌ Missing environment variable: {env_var}")

# API keys
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise RuntimeError("❌ Bị lỗi)

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

def transcribe_with_phowhisper(audio_path: str) -> str:
    """Nhận file audio và trả về transcript tiếng Việt"""
    result = asr_pipeline(audio_path, generate_kwargs={"language": "auto"})
    return result["text"]


    # 1. Chuyển file audio sang WAV 16-bit PCM nếu định dạng không chuẩn
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".wav", ".mp3", ".m4a", ".flac"]:
        print(f"🔄 Chuyển đổi {ext} → .wav")
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path + ".wav"
        audio.export(wav_path, format="wav")
        file_path = wav_path

    # 2. Upload theo từng chunk 5MB
    def read_file_in_chunks(path, chunk_size=6_291_456):
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

def groq_generate(prompt, max_tokens=1000, retries=3):
    """Gọi Groq API lần lượt từng key khi bị rate limit"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    # Thử từng key
    for key_index, api_key in enumerate(GROQ_API_KEYS):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        for attempt in range(retries):
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 429:
                wait_time = 2 ** attempt
                print(f"⚠️ Thử lại sau {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif res.status_code >= 500:
                break
            try:
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"⚠️ Key #{key_index+1} lỗi: {e}")
                break  # sang key tiếp theo

    raise Exception("❌ Bị lỗi")
    
def split_text(text, chunk_size=3000):
    """Chia văn bản thành các đoạn nhỏ"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
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
        language_code = request.form.get("language_code")
        language_name = request.form.get("language_name")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            ext = file.filename.lower()

            if ext.endswith((".mp3", ".wav")):
                text = transcribe_with_phowhisper(tmp.name, language_code)
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

        # === Xử lý Groq ===
        # Chủ đề
        subject = groq_generate(f"Hãy cho biết chủ đề chính của nội dung sau bằng {language_code}: {text[:4000]}")

        # Tóm tắt theo từng phần
        chunks = split_text(text, chunk_size=3000)
        partial_summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"🔹 Tóm tắt đoạn {idx+1}/{len(chunks)}")
            summary_part = groq_generate(
                f"Tóm tắt đoạn văn sau bằng {language_name}, ngắn gọn, đầy đủ ý:\n\n{chunk}",
                max_tokens=800
            )
            partial_summaries.append(summary_part)
            time.sleep(1)  # Giảm tải API

        # Tóm tắt cuối cùng từ các bản tóm tắt nhỏ
        final_summary = groq_generate(
            f"Dưới đây là các bản tóm tắt từng phần. Hãy gộp chúng thành một bản tóm tắt hoàn chỉnh, mạch lạc, bằng {language_name}:\n\n"
            + "\n\n".join(partial_summaries),
            max_tokens=1000
        )

        # Upload file gốc
        safe_file_name = f"uploads/{file.filename}"
        upload_to_b2(tmp.name, safe_file_name, content_type)
        file_url = get_signed_url(safe_file_name)

        # Upload JSON kết quả
        result_data = {
            "subject": subject,
            "summary": final_summary,
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
            "summary": final_summary,
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
        return jsonify({"error": "Thiếu"}), 400
    try:
        return jsonify({"signed_url": get_signed_url(file_name)})
    except Exception as e:
        return jsonify({"error": "Lỗi"}), 500

# API: Lấy nội dung JSON từ bucket private
@app.route("/get_json_content", methods=["GET"])
def get_json_content():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "Thiếu"}), 400
    try:
        signed_url = get_signed_url(file_name)
        res = requests.get(signed_url)
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": "Lỗi"}), 500
        
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    context = data.get("context", "")

    if not question:
        return jsonify({"error": "Thiếu câu hỏi"}), 400

    # Tạo prompt dựa trên transcript
    prompt = f"""
    Bạn là trợ lý AI.
    Đây là nội dung transcript/tóm tắt:
    {context}

    Câu hỏi: {question}
    Trả lời bằng {language_code}, rõ ràng, súc tích.
    """

    answer = groq_generate(prompt, max_tokens=800)

    return jsonify({"answer": answer})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



















