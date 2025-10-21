from flask import Flask, request, jsonify
import os, logging, tempfile, gc
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import easyocr
import google.generativeai as genai
from flask_cors import CORS

# ----------------------------------
# Setup
# ----------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

# ✅ Lazy load OCR model only once
ocr_reader = None

# ✅ Load Gemini config (safe)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDBfH2xSnbEQejRsjLGPlokpSGUIM0N4dA"))

# ----------------------------------
# OCR Extraction
# ----------------------------------
def extract_text(file_path: str) -> str:
    global ocr_reader
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            if ocr_reader is None:
                ocr_reader = easyocr.Reader(["en"], gpu=False)
            result = ocr_reader.readtext(file_path, detail=0)
            text = "\n".join(result) if result else "[No text]"
        else:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
    except Exception as e:
        logging.exception("OCR/PDF error:")
        return f"[Error extracting text: {e}]"
    finally:
        gc.collect()

    return text.strip()


# ----------------------------------
# Gemini Summary
# ----------------------------------
def summarize_with_gemini(text: str) -> str:
    if not text.strip():
        return "[No content to summarize]"
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash-lite")  # ✅ lighter model
        resp = model.generate_content(
            f"Summarize this medical report simply:\n\n{text[:4000]}"
        )
        return resp.text.strip() if hasattr(resp, "text") else "[No summary]"
    except Exception as e:
        logging.exception("Gemini API failed:")
        return f"[Gemini failed: {e}]"


# ----------------------------------
# Flask Route
# ----------------------------------
@app.route("/parse", methods=["POST"])
def parse():
    gc.collect()  # clean before processing

    # Case 1: Node sends file path
    if "file_path" in request.form:
        file_path = request.form["file_path"]
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 400

        text = extract_text(file_path)
        summary = summarize_with_gemini(text)
        gc.collect()
        return jsonify({"message": "Parsed (from path)", "summary": summary})

    # Case 2: Direct upload
    if "files" not in request.files:
        return jsonify({"error": "No files or file_path provided"}), 400

    files = request.files.getlist("files")
    combined_text = ""
    diagnostics = {}

    for f in files:
        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)

        text = extract_text(path)
        combined_text += f"\n=== {filename} ===\n{text}\n"
        diagnostics[filename] = {"length": len(text)}

        # delete file after reading to save memory
        os.remove(path)

    summary = summarize_with_gemini(combined_text)
    gc.collect()

    return jsonify({
        "message": "Parsed successfully",
        "diagnostics": diagnostics,
        "summary": summary
    })


# ----------------------------------
# Run App
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
