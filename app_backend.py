
# -*- coding: utf-8 -*-
"""app_backend.py: Production-Ready FastAPI Backend with Lifespan Management"""

# --- 0) Setup & Imports ---
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from pyngrok import ngrok
import uvicorn, time, os, json, pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler 
from contextlib import asynccontextmanager 
from datetime import datetime
import requests
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# --- 1) Configuration & Initialization ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
DAILY_CALL_HOUR_IST = 10

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
scheduler = AsyncIOScheduler() 
GLOBAL_MODEL, GLOBAL_EXPLAINER = None, None
PUBLIC_URL = "" 
PATIENTS_DB = {}
PATIENT_ID_COUNTER = 1001 

def get_next_patient_id():
    global PATIENT_ID_COUNTER
    new_id = f"P{PATIENT_ID_COUNTER}"
    PATIENT_ID_COUNTER += 1
    return new_id
# --- LIFESPAN: Start/Stop Scheduler and ML Model ---

# DUMMY/SYNTHETIC DATA GENERATION (Kept local for simplicity)
FEATURE_COLUMNS = ['age', 'prior_admissions_30d', 'comorbidity_score', 'adherence', 'sentiment_score_neg']
def generate_synthetic_data(num_samples=200):
    np.random.seed(42)
    data = {
        'age': np.random.randint(40, 95, num_samples),
        'prior_admissions_30d': np.random.randint(0, 4, num_samples),
        'comorbidity_score': np.random.randint(1, 8, num_samples),
        'adherence': np.random.randint(0, 2, num_samples),
        'sentiment_score_neg': np.random.uniform(0.1, 0.9, num_samples)
    }
    df = pd.DataFrame(data)
    risk = ((df['age'] > 70) * 0.3 + (df['prior_admissions_30d'] > 1) * 0.4 + (df['comorbidity_score'] > 4) * 0.2 + (df['adherence'] == 0) * 0.5 + (df['sentiment_score_neg'] > 0.6) * 0.3)
    prob = np.clip(risk / risk.max() + np.random.uniform(-0.1, 0.1, num_samples), 0, 1)
    df['readmission_status'] = (prob > 0.5).astype(int)
    return df

def train_ml_model():
    print("--- Training ML Model ---")
    data = generate_synthetic_data()
    X = data[FEATURE_COLUMNS]
    y = data['readmission_status']
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', learning_rate=0.1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    print("ML Model trained and SHAP explainer initialized.")
    return model, explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER, PUBLIC_URL
    
    # 1. Determine the Public URL based on the environment
    # RENDER_EXTERNAL_HOSTNAME is set automatically by Render for the public URL
    if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
        # We are on Render (Production)
        # PUBLIC_URL MUST use HTTPS and the hostname provided by Render
        PUBLIC_URL = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}"
        print(f"Deployment mode: Using Render URL {PUBLIC_URL}")
        
    else:
        # We are running locally (Development)
        try:
            # Conditional import of pyngrok to avoid issues on Render 
            # and use it for local development only.
            from pyngrok import ngrok 
            ngrok_tunnel = ngrok.connect(8000)
            PUBLIC_URL = ngrok_tunnel.public_url
            print(f"Development mode: Ngrok tunnel started at {PUBLIC_URL}")
        except Exception as e:
            # Fallback if ngrok fails locally (e.g., missing authtoken)
            print(f"Ngrok connection failed, proceeding with local URL. Error: {e}")
            PUBLIC_URL = "http://localhost:8000"

    print("\n" + "="*70)
    print("FastAPI Backend Starting...")

    # 2. Train Model
    GLOBAL_MODEL, GLOBAL_EXPLAINER = train_ml_model()

    # 3. Start Scheduler
    hour = DAILY_CALL_HOUR_IST
    scheduler.remove_all_jobs()
    scheduler.add_job(call_patients_job, trigger="cron", hour=hour, minute=0, timezone=pytz.timezone('Asia/Kolkata'))
    scheduler.start()
    print(f"Daily Scheduler set for {DAILY_CALL_HOUR_IST}:00 IST.")
    print("="*70 + "\n")

    yield # APPLICATION IS RUNNING HERE

    # 4. Stop Scheduler and Ngrok on Shutdown
    scheduler.shutdown()
    
    # Only try to clean up ngrok if we were running locally
    if not os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
        try:
            # Add cleanup logic if needed for local development
            print("Cleanup: Local environment shutdown complete.")
        except:
            pass 

    print("Backend shutting down.")


app = FastAPI(lifespan=lifespan)

# --- 2) CLINICAL KNOWLEDGE BASE & IVR HELPERS ---
CLINICAL_QUESTIONS = {
    "shortness of breath": ["Are you finding it hard to breathe when you are resting?", "Do you have to sit up to sleep because of your breathing?"],
    "swelling": ["Have your feet, ankles, or legs gotten bigger or more swollen?", "How much heavier do you feel in the last two days?"], 
    "cough": ["Are you coughing up any mucus or phlegm?", "What color is the mucus you are coughing up?"], 
    "dizziness": ["Did you feel very dizzy or faint after taking your blood pressure pills?", "Did you almost fall down or lose your balance?"],
    "furosemide": ["Did you take your water pill (Furosemide) today?", "Are you feeling very thirsty or having muscle pains after taking it?"],
    "lisinopril": ["Did you take your blood pressure pill (Lisinopril) today?", "Do you have a dry, constant cough?"],
    "salbutamol": ["Did you need to use your puff/inhaler (Salbutamol) more often today?", "Do you feel it is helping you breathe better?"],
}

CALL_STATES = {}
# (Keeping the rest of your helper functions: get_patient_context, make_symptom_item, enqueue_symptoms, clinical_ner_extract, get_contextual_questions, perform_nlp, predict_and_explain, perform_nlp_and_risk_update)

def get_patient_context(phone_number):
    for pid, context in PATIENTS_DB.items():
        if context.get("phone") == phone_number:
            return pid, context
    return None, None

def make_symptom_item(symptom_name):
    symptom_name = symptom_name.lower().strip()
    questions = CLINICAL_QUESTIONS.get(symptom_name, [f"Can you please tell us more about {symptom_name}?"])
    return {
        "name": symptom_name,
        "questions": questions,
        "q_index": 0,
        "answer_log": []
    }

def enqueue_symptoms(state, symptoms_list):
    existing_names = {item["name"] for item in state.get("symptoms", [])}
    for symptom in symptoms_list:
        if symptom not in existing_names:
            state["symptoms"].append(make_symptom_item(symptom))

def clinical_ner_extract(text):
    text_lower = text.lower()
    extracted = set()
    if any(word in text_lower for word in ['cough', 'mucus', 'phlegm', 'sputum']): extracted.add("cough")
    if any(word in text_lower for word in ['breath', 'wheez', 'gaspin', 'pant', 'gasp', 'shortness of breath']): extracted.add("shortness of breath")
    if any(word in text_lower for word in ['swell', 'ankle', 'feet', 'leg', 'fluid', 'heavy', 'tight shoes']): extracted.add("swelling")
    if any(word in text_lower for word in ['dizzy', 'light-head', 'faint', 'unstable', 'almost fall']): extracted.add("dizziness")
    return list(extracted)

def get_contextual_questions(patient_id, user_text):
    patient = PATIENTS_DB.get(patient_id)
    if not patient: return []
    explicit_symptoms = set(clinical_ner_extract(user_text))
    implied_symptoms = set()
    diagnosis = patient['discharge_summary']['diagnosis'].lower()
    medications = patient['discharge_summary']['medications'].lower()
    if 'chf' in diagnosis or 'heart failure' in diagnosis:
        implied_symptoms.update(["swelling", "shortness of breath"])
    if 'copd' in diagnosis or 'bronchitis' in diagnosis:
        implied_symptoms.update(["cough", "shortness of breath"])
    for key in CLINICAL_QUESTIONS.keys():
        if key.lower() in medications.lower(): 
            implied_symptoms.add(key)
    final_queue = list(explicit_symptoms)
    for s in implied_symptoms:
        if s not in explicit_symptoms:
            final_queue.append(s)
    return final_queue
    
def perform_nlp(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['worse', 'pain', 'barely breathe', 'critical']):
        return 0.90
    elif any(word in text_lower for word in ['fine', 'better', 'good', 'stable']):
        return 0.10
    else:
        return 0.50

def predict_and_explain(patient_data):
    features = {col: [patient_data.get(col)] for col in FEATURE_COLUMNS}
    X_single = pd.DataFrame(features)
    readmission_prob = GLOBAL_MODEL.predict_proba(X_single)[:, 1][0]
    shap_values = GLOBAL_EXPLAINER.shap_values(X_single)[1]
    explanation = sorted(zip(FEATURE_COLUMNS, shap_values), key=lambda x: abs(x[1]), reverse=True)
    formatted_explanation = [
        {"feature": name, "impact_score": round(score, 3)}
        for name, score in explanation[:5]
    ]
    return readmission_prob, formatted_explanation
    
def perform_nlp_and_risk_update(patient_id, symptoms_text, adherence_text):
    sentiment_score_neg = perform_nlp(symptoms_text)
    patient = PATIENTS_DB[patient_id]
    adherence_int = 1 if 'yes' in adherence_text.lower() else 0

    patient_features = {
        'age': patient['age'],
        'prior_admissions_30d': patient['prior_admissions_30d'],
        'comorbidity_score': patient['comorbidity_score'],
        'adherence': adherence_int,
        'sentiment_score_neg': sentiment_score_neg
    }

    risk_prob, shap_explanation = predict_and_explain(patient_features)

    if risk_prob >= 0.70: risk_level = "High"
    elif risk_prob >= 0.50: risk_level = "Medium"
    else: risk_level = "Low"

    new_report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adherence": adherence_int,
        "symptom_report": symptoms_text,
        "sentiment_score": sentiment_score_neg,
        "ml_prediction": round(risk_prob, 3),
        "risk_level": risk_level,
        "shap_explanation": shap_explanation,
    }

    if 'daily_reports_log' not in patient: patient['daily_reports_log'] = []

    patient['daily_reports_log'].insert(0, new_report) 
    patient.update(new_report)
    patient['last_call'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if risk_level == "High":
        print(f"\n🚨🚨 CRITICAL ALERT: Patient {patient['name']} (ID: {patient_id}) - RISK: {risk_level} ({risk_prob*100:.1f}%)")
    return patient

# --- 4) IVR Endpoints (Twilio Webhooks) ---
@app.post("/twilio")
async def twilio_webhook(request: Request, Caller: str = Form(None), CallSid: str = Form(None)):
    patient_id, patient = get_patient_context(Caller)
    if not patient_id:
        response = VoiceResponse()
        response.say("Sorry, we cannot find your file. Please call your care provider.")
        return Response(content=str(response), media_type="application/xml")
    CALL_STATES[CallSid] = {
        "id": patient_id, "symptoms": [], "current_symptom_index": 0, "symptoms_text_log": "",
        "adherence_text_log": "", "call_sid": CallSid
    }
    response = VoiceResponse()
    gather = response.gather(input='speech', action=f'/twilio/process_symptoms', timeout=10)
    gather.say(f"Hello {patient['name']}. This is your daily check-up. Please tell us briefly how you are feeling today. Mention anything new, like being sick or feeling worse.")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_symptoms")
async def process_initial_symptoms(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return Response(content=str(VoiceResponse().say("Session error. Goodbye.")), media_type="application/xml")
    symptoms_text = SpeechResult or "No symptoms reported"
    state['symptoms_text_log'] = symptoms_text
    patient_id = state['id']
    symptom_queue = get_contextual_questions(patient_id, symptoms_text)
    enqueue_symptoms(state, symptom_queue)
    response = VoiceResponse()
    gather = response.gather(input='speech', action=f'/twilio/process_adherence', timeout=5)
    gather.say("Thank you. Now, please tell us: did you take ALL of your pills and medicine today, exactly as the doctor told you? Please say 'Yes' or 'No'.")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_adherence")
async def process_adherence(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return Response(content=str(VoiceResponse().say("Session error. Goodbye.")), media_type="application/xml")
    adherence_response = SpeechResult or "No speech response"
    state['adherence_text_log'] = adherence_response
    response = VoiceResponse()
    if not state.get("symptoms"):
        response.say("Thank you for telling us about your medicine. Since you reported nothing new, your file is stable. Goodbye.")
        perform_nlp_and_risk_update(state['id'], state['symptoms_text_log'], state['adherence_text_log'])
        CALL_STATES.pop(CallSid, None)
        return Response(content=str(response), media_type="application/xml")
    current_symptom_item = state["symptoms"][state["current_symptom_index"]]
    current_q = current_symptom_item["questions"][current_symptom_item["q_index"]]
    gather = response.gather(input='speech', action=f'/twilio/process_answer', timeout=10)
    gather.say(f"Okay, let's ask a few questions about your {current_symptom_item['name']}. {current_q}")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_answer")
async def process_answer(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return Response(content=str(VoiceResponse().say("Session error. Goodbye.")), media_type="application/xml")
    patient_id = state['id']
    answer_text = SpeechResult or "No answer provided"
    response = VoiceResponse()
    current_idx = state.get("current_symptom_index", 0)
    if current_idx < len(state["symptoms"]):
        symptom_item = state["symptoms"][current_idx]
        current_q = symptom_item["questions"][symptom_item["q_index"]]
        symptom_item["answer_log"].append({"question": current_q, "answer": answer_text})
        symptom_item["q_index"] += 1
        if symptom_item["q_index"] < len(symptom_item["questions"]):
            next_q = symptom_item["questions"][symptom_item["q_index"]]
            gather = response.gather(input='speech', action=f'/twilio/process_answer', timeout=10)
            gather.say(f"Next question about {symptom_item['name']}: {next_q}")
            return Response(content=str(response), media_type="application/xml")
        state["current_symptom_index"] += 1
    if state["current_symptom_index"] < len(state["symptoms"]):
        next_item = state["symptoms"][state["current_symptom_index"]]
        next_q = next_item["questions"][next_item["q_index"]]
        gather = response.gather(input='speech', action=f'/twilio/process_answer', timeout=10)
        gather.say(f"Now, about your {next_item['name']}. {next_q}")
        return Response(content=str(response), media_type="application/xml")
    full_symptom_report = f"Initial: {state['symptoms_text_log']}"
    for item in state['symptoms']:
        for log in item['answer_log']:
            full_symptom_report += f" | {item['name']} Follow-up: {log['answer']}"
    updated_patient = perform_nlp_and_risk_update(
        patient_id, full_symptom_report, state['adherence_text_log']
    )
    response.say("Thank you for answering all the questions. Your information is now updated.")
    if updated_patient['risk_level'] == "High":
        response.say("Your doctor has been immediately alerted to review your report.")
    response.say("Goodbye.")
    CALL_STATES.pop(CallSid, None)
    return Response(content=str(response), media_type="application/xml")

# --- 5) Patient Enrollment Endpoints (API used by Streamlit) ---
@app.post("/api/patients/add")
async def add_new_patient_api(
    name: str = Form(...), phone: str = Form(...), age: int = Form(...),
    prior_admissions_30d: int = Form(...), comorbidity_score: int = Form(...),
    discharge_diagnosis: str = Form(...), medications: str = Form(...) 
):
    if not phone.startswith('+91') or not re.match(r'^\+91\d{10}$', phone):
        raise HTTPException(status_code=400, detail="Invalid Indian phone number format. Use E.164 starting with +91.")
    if any(p['phone'] == phone for p in PATIENTS_DB.values()):
        raise HTTPException(status_code=400, detail=f"Patient with phone {phone} already exists.")
    patient_id = get_next_patient_id()
    new_record = {
        "phone": phone, "name": name, "age": age, "prior_admissions_30d": prior_admissions_30d, 
        "comorbidity_score": comorbidity_score, "adherence": 1, "symptom_report": "New Patient Enrolled.",
        "sentiment_score": 0.0, "ml_prediction": 0.0, "risk_level": "Low", "shap_explanation": [],
        "last_call": None, "doctor_override": False, "intervention_notes": "",
        "discharge_summary": {
            "diagnosis": discharge_diagnosis, "date": datetime.now().strftime("%Y-%m-%d"),
            "medications": medications, "notes": "Patient enrolled post-discharge for 30-day monitoring."
        },
        "daily_reports_log": [],
    }
    PATIENTS_DB[patient_id] = new_record
    print(f"\n✅ New Patient Added: {name} (ID: {patient_id})")
    return {"status": "success", "patient_id": patient_id, "name": name}

# --- 6) Dashboard and Intervention Endpoints (APIs for Streamlit) ---
@app.get("/api/patients/all_summary")
async def get_all_patients_summary():
    summary = []
    for patient_id, p in PATIENTS_DB.items():
        summary.append({
            "id": patient_id, "name": p['name'], "risk_level": p['risk_level'],
            "risk_probability": p['ml_prediction'] * 100, "last_report": p['symptom_report'],
            "shap_explanation": p['shap_explanation'], "doctor_override": p['doctor_override'],
            "intervention_notes": p['intervention_notes'], "last_call": p['last_call']
        })
    summary.sort(key=lambda p: p['risk_probability'], reverse=True)
    return JSONResponse(content=summary)

@app.get("/api/patients/{patient_id}/history")
async def get_patient_history_api(patient_id: str):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = PATIENTS_DB[patient_id]
    response_data = {
        "patient_info": {"id": patient_id, "name": patient['name'], "age": patient['age'], "phone": patient['phone']},
        "discharge_summary": patient['discharge_summary'],
        "daily_data_log": patient.get('daily_reports_log', []),
    }
    return JSONResponse(content=response_data)

@app.post("/api/patients/intervene/{patient_id}")
async def log_intervention_api(patient_id: str, notes: Request):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    notes_json = await notes.json()
    intervention_notes = notes_json.get("notes", "No notes provided")
    PATIENTS_DB[patient_id]['doctor_override'] = True
    PATIENTS_DB[patient_id]['intervention_notes'] = intervention_notes
    print(f"\n👨‍⚕️ INTERVENTION LOGGED for {PATIENTS_DB[patient_id]['name']}")
    return {"status": "success", "patient_id": patient_id, "message": "Intervention logged."}


# --- 7) Scheduling and Server Control ---
def call_patient(patient_id, phone_number):
    global PUBLIC_URL
    try:
        # For deployment, this URL must be the public Render URL
        # For local test, it relies on PUBLIC_URL set by ngrok
        call = twilio_client.calls.create(
            to=phone_number, from_=TWILIO_NUMBER, url=f"{PUBLIC_URL}/twilio"
        )
        print(f"Call initiated for {patient_id}. SID: {call.sid}")
    except Exception as e:
        print(f"Error calling {patient_id}: {e}")

def call_patients_job():
    print(f"\n--- Running Call Job @ {datetime.now().strftime('%H:%M:%S')} ---")
    if not PATIENTS_DB:
        print("No patients enrolled in the system.")
        return
    for patient_id, patient in PATIENTS_DB.items():
        if patient.get('risk_level', 'Low') != 'High' and not patient.get('doctor_override', False):
            call_patient(patient_id, patient['phone'])
        else:
            print(f"Skipping automated call for {patient['name']}.")
    return {"status": "job_completed", "count": len(PATIENTS_DB)}

@app.post("/call_patients_job_manual")
async def manual_call_trigger():
    result = call_patients_job()
    return JSONResponse(content=result)

@app.get("/get_public_url")
async def get_public_url_api():
    """Endpoint for Streamlit to fetch the FastAPI URL"""

    return {"public_url": PUBLIC_URL}
