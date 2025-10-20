from flask import Flask, request, jsonify
import os, logging
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import easyocr
import google.generativeai as genai
from flask import Flask
from flask_cors import CORS
# ----------------- Setup -----------------
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
ocr_reader = easyocr.Reader(['en'])

API_KEY = "AIzaSyDBfH2xSnbEQejRsjLGPlokpSGUIM0N4dA"
genai.configure(api_key=API_KEY)

# ----------------- OCR -----------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        res = ocr_reader.readtext(file_path, detail=0)
        text = "\n".join(res) if res else "[No text]"
    else:  # PDF
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    return text

# ----------------- Gemini Analysis -----------------
def summarize_with_gemini(text):
    prompt = f"""
You are a medical assistant. Summarize key points of this report in simple terms:

{text[:5000]}
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, "text") else "[No summary]"
    except Exception as e:
        logging.exception("Gemini failed:")
        return f"Gemini failed: {str(e)}"

# ----------------- Route -----------------
@app.route('/parse', methods=['POST'])
def parse():
    # 🔹 Case 1: Node sends file path
    if 'file_path' in request.form:
        file_path = request.form['file_path']
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 400
        text = extract_text(file_path)
        summary = summarize_with_gemini(text)
        return jsonify({
            "message": "Parsed successfully (from path)",
            "file": file_path,
            "summary": summary
        })

    # 🔹 Case 2: Client directly uploads files
    if 'files' not in request.files:
        return jsonify({"error": "No files or file_path provided"}), 400

    files = request.files.getlist('files')
    combined_text = ""
    diagnostics = {}

    for f in files:
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        try:
            txt = extract_text(path)
            combined_text += f"\n=== {filename} ===\n{txt}\n"
            diagnostics[filename] = {"text_length": len(txt)}
        except Exception as e:
            diagnostics[filename] = {"error": str(e)}

    summary = summarize_with_gemini(combined_text)
    return jsonify({
        "message": "Parsed successfully (uploaded)",
        "diagnostics": diagnostics,
        "summary": summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
