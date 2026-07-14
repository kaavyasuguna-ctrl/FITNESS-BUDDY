import os
import time
import uuid
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-dev-secret-key")

# ---------------------------------------------------------------------------
# Chat history store  { session_id: {"messages": [...], "expires_at": float} }
# Each session expires 5 hours after the last message.
# ---------------------------------------------------------------------------
CHAT_HISTORY: dict = {}
CHAT_TTL_SECONDS = 5 * 60 * 60  # 5 hours


def _prune_expired():
    """Remove sessions whose TTL has elapsed."""
    now = time.time()
    expired = [sid for sid, v in CHAT_HISTORY.items() if v["expires_at"] < now]
    for sid in expired:
        del CHAT_HISTORY[sid]


def get_session(session_id: str) -> list:
    _prune_expired()
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = {"messages": [], "expires_at": 0}
    entry = CHAT_HISTORY[session_id]
    entry["expires_at"] = time.time() + CHAT_TTL_SECONDS
    return entry["messages"]

# ---------------------------------------------------------------------------
# Watsonx.ai client setup
# ---------------------------------------------------------------------------
def get_watsonx_model():
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    api_key     = os.getenv("IBM_API_KEY")
    project_id  = os.getenv("WATSONX_PROJECT_ID")

    if not api_key:
        raise ValueError("IBM_API_KEY is not set in your .env file.")
    if not project_id:
        raise ValueError("WATSONX_PROJECT_ID is not set in your .env file.")
    if "dataplatform.cloud.ibm.com" in watsonx_url:
        raise ValueError(
            "WATSONX_URL is set to the Watson Studio web UI (dataplatform.cloud.ibm.com). "
            "Use the inference API endpoint instead, e.g. https://us-south.ml.cloud.ibm.com"
        )

    is_saas = "ml.cloud.ibm.com" in watsonx_url
    logging.debug("Connecting to Watsonx at: %s (SaaS=%s)", watsonx_url, is_saas)

    if is_saas:
        # IBM Cloud SaaS — api_key + IAM token exchange
        credentials = Credentials(url=watsonx_url, api_key=api_key)
    else:
        # Cloud Pak for Data on-prem — requires username + api_key + version
        cpd_username = os.getenv("CPD_USERNAME", "apikey")
        cpd_version  = os.getenv("CPD_VERSION", "5.0")
        credentials = Credentials(
            url=watsonx_url,
            username=cpd_username,
            api_key=api_key,
            version=cpd_version,
        )
    client = APIClient(credentials)
    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_client=client,
        project_id=project_id,
        params={
            GenParams.MAX_NEW_TOKENS: 900,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.9,
            GenParams.REPETITION_PENALTY: 1.1,
        },
    )
    return model


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(age, location, lifestyle, weight, fitness_level,
                 health_conditions, allergies):
    has_alert = bool(
        (allergies and allergies.strip().lower() not in ("none", "n/a", ""))
        or (health_conditions and health_conditions.strip().lower() not in ("none", "n/a", ""))
    )

    alert_instruction = ""
    if has_alert:
        alert_instruction = (
            "IMPORTANT: Print '⚠️ CRITICAL MEDICAL ALERT' as the very first line, "
            "then list all allergens to BLOCK and explain exercise modifications needed. "
        )

    prompt = f"""<|system|>
You are FitBuddy AI, a certified fitness and nutrition advisor. You give personalised, safe, and practical wellness plans.
{alert_instruction}Always respond in the exact structured format shown below. Be concise but thorough.
<|user|>
Profile: Age={age} | Location={location} | Lifestyle={lifestyle} | Weight={weight}kg | Fitness Level={fitness_level} | Health Conditions={health_conditions} | Allergies={allergies}

Generate a complete wellness plan using EXACTLY this format:

{('⚠️ CRITICAL MEDICAL ALERT' + chr(10) + '- Blocked allergens: list them' + chr(10) + '- Medical adaptations: list them' + chr(10) if has_alert else '')}
**Core Fitness Habits:**
- [Habit 1 tailored to age and lifestyle]
- [Habit 2 for daily routine]
- [Habit 3 for consistency]

**Nutrition & Hydration:**
Diet: [Personalised meal guide for energy and health conditions]
Water: [Daily liters target based on {location} climate and {lifestyle} workload]
ALERT: [If applicable — Avoid specific allergens. Manage specific health conditions. Otherwise write: None.]

**Exercise & Yoga:**
Level: [Fitness level label based on {fitness_level} and age {age}]
Routine:
1. [Exercise/pose name] — [brief description and duration]
2. [Exercise/pose name] — [brief description and duration]
3. [Exercise/pose name] — [brief description and duration]
PRECAUTION: [If applicable — joint/surgery/breathing modifications. Otherwise write: None.]
<|assistant|>
"""
    return prompt


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    age             = data.get("age", "").strip()
    location        = data.get("location", "").strip()
    lifestyle       = data.get("lifestyle", "").strip()
    weight          = data.get("weight", "").strip()
    fitness_level   = data.get("fitness_level", "").strip()
    health_conditions = data.get("health_conditions", "None").strip()
    allergies       = data.get("allergies", "None").strip()

    # Basic validation
    required = {"age": age, "location": location, "lifestyle": lifestyle,
                "weight": weight, "fitness_level": fitness_level}
    missing = [k for k, v in required.items() if not v]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        model  = get_watsonx_model()
        prompt = build_prompt(age, location, lifestyle, weight,
                              fitness_level, health_conditions, allergies)
        response = model.generate_text(prompt=prompt)
        plan_text = response.strip() if isinstance(response, str) else response
        return jsonify({"plan": plan_text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    user_message = (data.get("message") or "").strip()
    session_id   = (data.get("session_id") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required."}), 400

    # Create a new session id if the client doesn't have one yet
    if not session_id:
        session_id = str(uuid.uuid4())

    history = get_session(session_id)
    history.append({"role": "user", "content": user_message})

    # Build conversation prompt
    system = (
        "You are FitBuddy AI, a friendly and knowledgeable fitness and nutrition assistant. "
        "Answer concisely and helpfully. If a question is unrelated to fitness, nutrition, "
        "or wellness, politely redirect the user."
    )
    turns = ""
    for msg in history:
        if msg["role"] == "user":
            turns += f"<|user|>\n{msg['content']}\n"
        else:
            turns += f"<|assistant|>\n{msg['content']}\n"
    prompt = f"<|system|>\n{system}\n{turns}<|assistant|>\n"

    try:
        model  = get_watsonx_model()
        reply  = model.generate_text(prompt=prompt)
        reply  = reply.strip() if isinstance(reply, str) else str(reply)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    history.append({"role": "assistant", "content": reply})

    return jsonify({
        "reply":      reply,
        "session_id": session_id,
        "history":    history,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
