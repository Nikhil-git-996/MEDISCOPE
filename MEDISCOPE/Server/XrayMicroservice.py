import os
import json
import logging
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ExifTags
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
import requests

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImageService")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# ----------------- Config -----------------
MODEL_URL = "https://huggingface.co/Nikhil2104/MEDISCOPE/resolve/main/final_best_model.keras"
MODEL_PATH = "final_best_model.keras"
IMG_SIZE = (160, 160)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# ----------------- Download Model -----------------
if not os.path.exists(MODEL_PATH):
    logger.info("⏳ Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    logger.info("✅ Model downloaded successfully.")

# ----------------- Load Model -----------------
logger.info("⏳ Loading ML model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logger.info("✅ Model loaded successfully.")

class_names = [
    "Brain_Hemorrhage", "Brain_Normal", "Brain_Tumor",
    "Chest_PNEUMONIA", "Chest_Xray_Normal",
    "Kidney_Normal", "Kidney_Stone", "Kidney_Tumor",
    "Knee_Normal", "Knee_Osteoarthritis",
    "Lung_Cancer", "Lungs_Normal", "Lungs_TB", "No_Lung_Cancer"
]

# ----------------- Helpers -----------------
def allowed_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def correct_exif_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = img._getexif()
        if exif is None:
            return img
        orientation_value = exif.get(orientation, None)
        if orientation_value == 3:
            img = img.rotate(180, expand=True)
        elif orientation_value == 6:
            img = img.rotate(270, expand=True)
        elif orientation_value == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def load_and_prepare_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = correct_exif_orientation(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    w, h = img.size
    if min(w, h) < 32:
        img = img.resize((max(w, 32), max(h, 32)), Image.BICUBIC)
        w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    img = img.crop((left, top, right, bottom))
    img = img.resize(IMG_SIZE, Image.BICUBIC)
    return img

def pil_to_model_array(pil_img):
    arr = np.array(pil_img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr]*3, axis=-1) if arr.ndim == 2 else arr[:, :, :3]
    arr = arr.astype(np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

def predict_with_tta(img):
    base = load_and_prepare_image(img)
    imgs = [base, ImageOps.mirror(base)]
    arrays = np.stack([pil_to_model_array(im) for im in imgs], axis=0)
    preds = model.predict(arrays, verbose=0)
    avg = np.mean(preds, axis=0)
    class_idx = int(np.argmax(avg))
    confidence = float(avg[class_idx])
    return class_names[class_idx], confidence

def decode_base64_image(data):
    try:
        image_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError("Invalid base64 image data")

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "payload" not in data:
        return jsonify({"error": "Missing payload"}), 400

    payload = data["payload"]
    img = None
    if "image_base64" in payload:
        try:
            img = decode_base64_image(payload["image_base64"])
        except Exception as e:
            return jsonify({"error": "Invalid base64 image", "message": str(e)}), 400
    elif "image_path" in payload:
        if not os.path.exists(payload["image_path"]):
            return jsonify({"error": "File not found", "path": payload["image_path"]}), 400
        img = Image.open(payload["image_path"]).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    age = payload.get("age")
    weight = payload.get("weight")
    symptoms = payload.get("symptoms", "")
    body_part = payload.get("body_part", "")

    try:
        pred_class, confidence = predict_with_tta(img)
    except Exception as e:
        logger.exception("Prediction error:")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500

    result = {
        "prediction": {
            "class": pred_class,
            "confidence": confidence
        },
        "patient_info": {
            "age": age,
            "weight": weight,
            "symptoms": symptoms,
            "body_part": body_part
        }
    }
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
