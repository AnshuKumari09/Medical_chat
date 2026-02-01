# Medical Chat
AI Medical Chatbot — Groq LLaMA 3.3 + Flask

## Setup
```bash
python -m venv medical_env
medical_env\Scripts\activate
pip install -r requirements.txt
```
Create `.env` from `.env.example`, paste your GROQ_API key.
```bash
python app4.py
```
Open: http://localhost:5000

## API
```
POST /api/chat
{ "message": "What is diabetes?" }
→ { "reply": "..." }
```

## Deploy → Render
1. Push this repo to GitHub
2. Render → New → Web Service → connect repo
3. Add env var: GROQ_API = your key
4. Deploy
