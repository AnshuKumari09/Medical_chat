from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)
CORS(app)  # Flutter ke liye zaroori — har origin allow karta hai

# ============================================================================
# AI MODEL — unchanged, same as tumhara
# ============================================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile"
)

system_prompt = """You are a medical information assistant. Follow these rules STRICTLY:

CRITICAL SAFETY RULES:
1. 🚨 EMERGENCIES: If user mentions chest pain, difficulty breathing, severe bleeding, stroke symptoms, 
   loss of consciousness, or any emergency symptoms, IMMEDIATELY say:
   "⚠️ This is a medical emergency! Call emergency services (911/108) IMMEDIATELY!"

2. 🚫 BOUNDARIES:
   - You CANNOT diagnose diseases
   - You CANNOT prescribe medications
   - You CANNOT replace a doctor consultation
   - Always say "Please consult a doctor for proper diagnosis and treatment"

3. 📋 ICD CODES: If asked for ICD-10 codes, provide them if you know:
   - E11.9 = Type 2 Diabetes Mellitus
   - E10.9 = Type 1 Diabetes Mellitus
   - I10 = Essential Hypertension
   - J45.9 = Asthma
   - J18.9 = Pneumonia
   - J44.9 = COPD
   - U07.1 = COVID-19
   If you don't know a code, say "I don't have that specific ICD code, please consult medical records."

4. ⚠️ MISINFORMATION: Never claim diseases can be "cured" with herbs/home remedies alone.
   Never tell patients to stop prescribed medications.

5. 🤝 BE HELPFUL: Provide general health information, explain medical terms in simple language,
   suggest when to see a doctor, and be supportive.

6. 🌐 LANGUAGE: Support both English and Hindi. Respond in the language user prefers.

7. ❓ VAGUE QUESTIONS: If question is unclear, ask for more details before answering.

Remember: You provide INFORMATION only, not medical advice or diagnosis.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()


# ============================================================================
# ROUTES
# ============================================================================

# Tumhara purana HTML route — ab bhi kaam karega agar local testing chahiye
@app.route("/")
def home():
    return render_template("index.html")


# Purana endpoint — tumhare HTML form ke liye (unchanged)
@app.route("/get", methods=["POST"])
def chat_form():
    user_msg = request.form.get("msg", "")
    if not user_msg:
        return "No message received", 400
    try:
        reply = chain.invoke({"question": user_msg})
        return reply
    except Exception as e:
        return f"Error: {str(e)}", 500


# ─── NEW: Flutter ke liye JSON endpoint ──────────────────────────────────────
# Flutter yahan POST karega:  { "message": "meri sar dard ho rahi hai" }
# Response milega:            { "reply": "..." }
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat_api():
    # JSON body parse karo
    data = request.get_json(silent=True)

    if not data or "message" not in data:
        return jsonify({"error": "Send JSON with 'message' field"}), 400

    user_msg = data["message"].strip()

    if not user_msg:
        return jsonify({"error": "Message is empty"}), 400

    try:
        reply = chain.invoke({"question": user_msg})
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# RUN
# ============================================================================
# host=0.0.0.0 → network pe accessible hoga (sirf localhost nahi)
# port=5000    → default Flask port
# ============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)