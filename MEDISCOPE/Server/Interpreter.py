from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from googletrans import Translator
from dotenv import load_dotenv
import logging
import json
import os

# ----------------------------------
# Flask + Logging Setup
# ----------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------
# Load environment variables
# ----------------------------------
load_dotenv()
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("API_KEY="):
                    API_KEY = line.strip().split("=", 1)[1]
                    logging.info("✅ Loaded API_KEY from .env fallback.")
                    break
    except FileNotFoundError:
        logging.warning("⚠️ .env not found for fallback loading.")

if not API_KEY:
    logging.error("❌ Missing API_KEY.")
    raise Exception("Missing API_KEY. Please set it in environment or .env file.")

# Configure Gemini
genai.configure(api_key=API_KEY)
logging.info("✅ Gemini API configured successfully.")

# ----------------------------------
# Translator + Language Mapping
# ----------------------------------
translator = Translator()
LANGUAGE_MAP = {"english": "en", "telugu": "te", "hindi": "hi"}

# ----------------------------------
# Core Response Generator
# ----------------------------------
def generate_health_response(username, content, mode="report", language="english"):
    # --- Custom greeting (only once) ---
    greeting = f"👋 Hello {username}," if username and username.lower() not in ["patient", "none", ""] else "👋 Hello there,"

    # --- Build prompt ---
    if mode == "report":
        prompt = f"""
You are a compassionate medical professional explaining a diagnostic or radiology report.

Report Data:
-------------------------
{content}
-------------------------

Your task:
1. Do NOT begin with any greeting or name (the system adds that already).
2. Explain findings clearly and kindly in layman terms.
3. Say if results appear normal or may require attention.
4. If abnormal, suggest what kind of **specialist** to see.
5. Give 3–5 brief **home-care and lifestyle tips** (rest 😴, hydration 💧, warm compress 🌿, fresh diet 🍎, etc.).
6. Encourage professional consultation for confirmation.
7. Keep it under 10 lines, friendly but professional (💊🩺😊).
"""
    else:
        prompt = f"""
You are a friendly medical assistant giving conversational wellness guidance.

The patient says:
-------------------------
{content}
-------------------------

Respond with:
- No greeting at the start.
- Simple, clear suggestions (no jargon).
- 2–3 easy home remedies (hydration 💧, rest 😴, herbal tea 🌿).
- Mention which doctor to consult if needed.
- Keep under 6 lines, positive and reassuring.
"""

    # --- Model configuration ---
    primary_model = "models/gemini-2.5-flash"
    fallback_model = "models/gemini-1.5-pro-latest"

    try:
        logging.info(f"🧠 Generating medical response using {primary_model} | Mode: {mode}")
        model = genai.GenerativeModel(primary_model)
        response = model.generate_content(prompt)
        final_response = response.text.strip() if hasattr(response, "text") and response.text else ""
        if not final_response:
            raise ValueError("Empty response from primary model.")
    except Exception as e:
        logging.error(f"⚠️ Primary model failed: {e}. Trying fallback model...")
        try:
            fallback = genai.GenerativeModel(fallback_model)
            response = fallback.generate_content(prompt)
            final_response = response.text.strip() if hasattr(response, "text") and response.text else ""
        except Exception:
            final_response = "Sorry, I'm having trouble generating a response right now. Please try again later."

    # --- Remove any duplicate greetings from model output ---
    import re
    final_response = re.sub(r"(?i)^\s*(hi|hello|hey)[^a-zA-Z]*", "", final_response).strip()
    final_response = re.sub(r"(?i)(^|\n)\s*(hi|hello|hey|dear)[^\n]*\n?", "", final_response).strip()

    # --- Translation ---
    if language.lower() != "english":
        dest_lang = LANGUAGE_MAP.get(language.lower(), "en")
        try:
            translated = translator.translate(final_response, dest=dest_lang)
            final_response = translated.text
        except Exception:
            final_response += "\n\n(Note: Translation to your selected language failed.)"

    return f"{greeting}\n\n{final_response}"

# ----------------------------------
# API Endpoint: /interpret
# ----------------------------------
@app.route("/interpret", methods=["POST"])
def interpret():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        username = data.get("username", "Patient")
        language = data.get("language", "english")
        mode = data.get("type", "report").lower()

        if mode == "chat":
            content = data.get("query", "")
            if not content:
                logging.error("❌ Missing 'query' field for chat request.")
                return jsonify({"error": "Missing 'query' for chat type"}), 400
        else:
            raw_preds = data.get("predictions", "{}")
            try:
                preds_json = json.loads(raw_preds) if isinstance(raw_preds, str) else raw_preds
                content = json.dumps(preds_json, indent=2)
            except json.JSONDecodeError:
                logging.warning("⚠️ Invalid JSON in predictions; using raw text.")
                content = raw_preds or "No diagnostic data provided."

        response_text = generate_health_response(username, content, mode, language)
        logging.info("✅ Successfully generated interpreter response.")
        return jsonify({"response": response_text})

    except Exception as e:
        logging.exception("Error in /interpret route:")
        return jsonify({"error": str(e)}), 500


# ----------------------------------
# Health Check Endpoint
# ----------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🩺 AI Health Interpreter (Dr. Athena) is active and ready.",
        "usage": {
            "POST /interpret": {
                "fields": {
                    "username": "string (optional)",
                    "language": "english/telugu/hindi",
                    "type": "report/chat",
                    "predictions": "JSON (for report)",
                    "query": "string (for chat)"
                }
            }
        }
    })

# ----------------------------------
# Run App
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
