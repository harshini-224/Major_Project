# -*- coding: utf-8 -*-
"""app_backend.py: Production-Ready FastAPI Backend with Persistent SQLite DB"""

# --- 0) Setup & Imports ---
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import uvicorn, time, os, json, pytz
import sqlite3 # <--- NEW: For persistent storage
from apscheduler.schedulers.asyncio import AsyncIOScheduler 
from contextlib import asynccontextmanager 
from datetime import datetime, timedelta
import requests
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# --- 1) Configuration & Initialization ---
# Fetch credentials from environment variables
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
DAILY_CALL_HOUR_IST = 10

# Database Configuration
DB_FILE = "patient_monitoring.db" # SQLite file for persistent storage
GLOBAL_DB_CONNECTION = None
PATIENT_ID_COUNTER = 1001 
GLOBAL_MODEL, GLOBAL_EXPLAINER = None, None
PUBLIC_URL = "" 

# In-memory session tracking for IVR calls (still needed for active calls)
CALL_STATES = {}

# Initialize client and scheduler
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Warning: Twilio client initialization failed at import time. Error: {e}")
    twilio_client = None

scheduler = AsyncIOScheduler() 

def get_next_patient_id():
    global PATIENT_ID_COUNTER
    new_id = f"P{PATIENT_ID_COUNTER}"
    PATIENT_ID_COUNTER += 1
    return new_id

# --- DATABASE HELPERS ---
def initialize_database():
    """Establishes connection and creates the patient table if it doesn't exist."""
    global GLOBAL_DB_CONNECTION, PATIENT_ID_COUNTER
    try:
        # check_same_thread=False is necessary for FastAPI's async/threading
        GLOBAL_DB_CONNECTION = sqlite3.connect(DB_FILE, check_same_thread=False)
        cursor = GLOBAL_DB_CONNECTION.cursor()

        # Schema: Complex objects stored as JSON strings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                phone TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                prior_admissions_30d INTEGER,
                comorbidity_score INTEGER,
                doctor_override INTEGER,
                intervention_notes TEXT,
                
                discharge_summary_json TEXT,
                current_report_json TEXT, 
                daily_reports_log_json TEXT 
            );
        """)
        
        # Initialize PATIENT_ID_COUNTER based on existing data
        cursor.execute("SELECT MAX(CAST(SUBSTR(id, 2) AS INTEGER)) FROM patients")
        max_id = cursor.fetchone()[0]
        if max_id:
            PATIENT_ID_COUNTER = max_id + 1
            print(f"Database found. Next patient ID set to: P{PATIENT_ID_COUNTER}")
        
        GLOBAL_DB_CONNECTION.commit()
        print(f"SQLite database initialized at {DB_FILE}")
        return GLOBAL_DB_CONNECTION
    except Exception as e:
        print(f"FATAL DB ERROR: {e}")
        return None

def db_to_patient_dict(row, cursor_description):
    """Converts a SQLite row tuple into the full Python dictionary structure."""
    if not row: return None
    
    patient_dict = {}
    col_names = [col[0] for col in cursor_description]
    for name, value in zip(col_names, row):
        if name.endswith('_json'):
            patient_dict[name.replace('_json', '')] = json.loads(value) if value else {}
        else:
            patient_dict[name] = value

    # Extract current risk/adherence fields from current_report for direct access
    current_report = patient_dict.get('current_report', {})
    patient_dict.update({
        'adherence': current_report.get('adherence', 1),
        'symptom_report': current_report.get('symptom_report', 'New Patient Enrolled.'),
        'sentiment_score': current_report.get('sentiment_score', 0.0),
        'ml_prediction': current_report.get('ml_prediction', 0.0),
        'risk_level': current_report.get('risk_level', 'Low'),
        'shap_explanation': current_report.get('shap_explanation', []),
        'last_call': current_report.get('date', None),
    })
    
    return patient_dict

def get_patient_context(identifier, by_phone=True):
    """Fetches patient data from SQLite by phone number or ID."""
    conn = GLOBAL_DB_CONNECTION
    if not conn: return None, None
    cursor = conn.cursor()
    
    if by_phone:
        query = "SELECT * FROM patients WHERE phone = ?"
    else:
        query = "SELECT * FROM patients WHERE id = ?"
        
    cursor.execute(query, (identifier,))
    row = cursor.fetchone()
    
    if row:
        patient_id = row[0]
        patient_dict = db_to_patient_dict(row, cursor.description)
        return patient_id, patient_dict
    return None, None

def save_new_patient_record(patient_id, new_record):
    """Inserts a new patient record into the database."""
    conn = GLOBAL_DB_CONNECTION
    if not conn: return False
    
    cursor = conn.cursor()
    
    # Serialize complex objects to JSON strings
    current_report_json = json.dumps({
        'adherence': new_record.get('adherence'), 'symptom_report': new_record.get('symptom_report'),
        'sentiment_score': new_record.get('sentiment_score'), 'ml_prediction': new_record.get('ml_prediction'),
        'risk_level': new_record.get('risk_level'), 'shap_explanation': new_record.get('shap_explanation', []),
        'date': new_record.get('last_call')
    })
    
    sql = """INSERT INTO patients (
        id, phone, name, age, prior_admissions_30d, comorbidity_score, doctor_override, intervention_notes, 
        discharge_summary_json, current_report_json, daily_reports_log_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    
    try:
        cursor.execute(sql, (
            patient_id, new_record['phone'], new_record['name'], new_record['age'], new_record['prior_admissions_30d'], 
            new_record['comorbidity_score'], int(new_record['doctor_override']), new_record['intervention_notes'], 
            json.dumps(new_record['discharge_summary']), current_report_json, json.dumps(new_record['daily_reports_log'])
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Patient {patient_id} (phone {new_record['phone']}) already exists.")
        return False
    except Exception as e:
        print(f"DB Insert Error: {e}")
        return False

def update_patient_fields(patient_id, updated_patient_dict):
    """Updates the dynamic fields (reports, risk, adherence, intervention) for an existing patient."""
    conn = GLOBAL_DB_CONNECTION
    if not conn: return False
    
    cursor = conn.cursor()
    
    # Prepare the latest report for the dedicated current_report_json field
    latest_report = updated_patient_dict['daily_reports_log'][0] if updated_patient_dict.get('daily_reports_log') else {}

    sql = """UPDATE patients SET
        doctor_override = ?,
        intervention_notes = ?,
        current_report_json = ?,
        daily_reports_log_json = ?
        WHERE id = ?"""
    
    try:
        cursor.execute(sql, (
            int(updated_patient_dict['doctor_override']),
            updated_patient_dict['intervention_notes'],
            json.dumps(latest_report), 
            json.dumps(updated_patient_dict['daily_reports_log']),
            patient_id
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"DB Update Error for {patient_id}: {e}")
        return False
        
def get_all_patients_data():
    """Fetches all patients from the database for the dashboard summary."""
    conn = GLOBAL_DB_CONNECTION
    if not conn: return []
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM patients")
    rows = cursor.fetchall()
    
    all_patients = []
    for row in rows:
        all_patients.append(db_to_patient_dict(row, cursor.description))
        
    return all_patients
    
def generate_historical_report(day_offset, adherence, symptoms, sentiment_score, ml_prob, risk_level):
    """Generates a single historical report entry for sample data."""
    base_date = datetime.now().date()
    report_date = datetime(base_date.year, base_date.month, base_date.day) - timedelta(days=day_offset)
    
    explanation = [
        {"feature": "sentiment_score_neg", "impact_score": round(sentiment_score * 0.4, 3)},
        {"feature": "adherence", "impact_score": round(0.3 if adherence == 0 else -0.1, 3)},
    ]

    return {
        "date": report_date.strftime("%Y-%m-%d 10:00:00"),
        "adherence": adherence,
        "symptom_report": symptoms,
        "sentiment_score": sentiment_score,
        "ml_prediction": round(ml_prob, 3),
        "risk_level": risk_level,
        "shap_explanation": explanation,
    }

def load_sample_patients():
    """Loads sample data only if the database is currently empty."""
    global PATIENT_ID_COUNTER
    if get_all_patients_data():
        print("Database already contains patient data. Skipping sample data load.")
        return
        
    print("Loading sample patient data with 14-day history...")

    # --- History for Pooja Singh (P1003): Risk slowly rising over 14 days ---
    history = []
    # Days 14 to 8: Low Risk
    history.extend([generate_historical_report(day, 1, "Feeling fine.", 0.1, 0.25, "Low") for day in range(8, 15)])
    # Day 7: Medium Risk
    history.append(generate_historical_report(7, 1, "Slight swelling, nothing major.", 0.5, 0.55, "Medium"))
    # Days 6 to 3: Medium Risk, Poor Adherence
    history.extend([generate_historical_report(day, 0, f"Minor cough on day {day}.", 0.6, 0.60, "Medium") for day in range(3, 7)])
    # Day 2: High Risk, Critical Symptoms Reported
    history.append(generate_historical_report(2, 0, "Hard to breathe, severe swelling in ankles.", 0.85, 0.78, "High"))
    # Day 1: High Risk (Recent Intervention)
    history.append(generate_historical_report(1, 1, "Doctor called, following new dosage. Still cautious.", 0.70, 0.75, "High"))
    
    history.sort(key=lambda x: x['date'], reverse=True)
    current_report = history[0]

    sample_data_list = [
        # --- Stable Patient ---
        {"id": "P1001", "phone": "+919900000001", "name": "Rajesh Kumar", "age": 75, "prior_admissions_30d": 2, 
        "comorbidity_score": 6, "adherence": 1, "symptom_report": "Patient feels stable.", "sentiment_score": 0.1, "ml_prediction": 0.350, "risk_level": "Low", 
        "shap_explanation": [{"feature": "age", "impact_score": 0.20}], "last_call": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "doctor_override": False, "intervention_notes": "", "discharge_summary": {"diagnosis": "CHF", "date": "2025-11-20", "medications": "Furosemide, Lisinopril", "notes": ""},
        "daily_reports_log": []},
        
        # --- Patient with Evolving Risk (14-day history) ---
        {"id": "P1003", "phone": "+919900000003", "name": "Pooja Singh", "age": 68, "prior_admissions_30d": 1, 
        "comorbidity_score": 4, "adherence": current_report['adherence'], "symptom_report": current_report['symptom_report'],
        "sentiment_score": current_report['sentiment_score'], "ml_prediction": current_report['ml_prediction'], 
        "risk_level": current_report['risk_level'], "shap_explanation": current_report['shap_explanation'],
        "last_call": current_report['date'], "doctor_override": True, 
        "intervention_notes": "Alerted on Day 2 due to severe edema. New diuretic prescribed on Day 1.",
        "discharge_summary": {"diagnosis": "Hypertensive Crisis", "date": "2025-11-15", "medications": "Lisinopril, Furosemide", "notes": ""},
        "daily_reports_log": history}
    ]
    
    for record in sample_data_list:
        save_new_patient_record(record['id'], record)
        
    PATIENT_ID_COUNTER = 1004
    print(f"Loaded {len(sample_data_list)} sample patients into SQLite.")


# --- LIFESPAN: Start/Stop Scheduler and ML Model ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER, PUBLIC_URL, twilio_client, GLOBAL_DB_CONNECTION
    ngrok_tunnel = None 

    if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
        PUBLIC_URL = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}"
    else:
        try:
            from pyngrok import ngrok 
            ngrok_tunnel = ngrok.connect(8000)
            PUBLIC_URL = ngrok_tunnel.public_url
        except Exception:
            PUBLIC_URL = "http://localhost:8000"

    print("\n" + "="*70)
    print("FastAPI Backend Starting...")
    
    # NEW STEP: Initialize Database and load samples
    GLOBAL_DB_CONNECTION = initialize_database()
    if not GLOBAL_DB_CONNECTION:
        raise RuntimeError("Failed to initialize database connection.")
        
    load_sample_patients() # Load samples only if DB is empty

    # 2. Train Model
    GLOBAL_MODEL, GLOBAL_EXPLAINER = train_ml_model()

    # 3. Start Scheduler
    hour = DAILY_CALL_HOUR_IST
    scheduler.remove_all_jobs()
    scheduler.add_job(call_patients_job, trigger="cron", hour=hour, minute=0, timezone=pytz.timezone('Asia/Kolkata'))
    scheduler.start()
    print(f"Daily Scheduler set for {DAILY_CALL_HOUR_IST}:00 IST.")
    print("="*70 + "\n")

    yield 

    # 4. Stop Scheduler and Clean up
    scheduler.shutdown()
    if GLOBAL_DB_CONNECTION:
        GLOBAL_DB_CONNECTION.close() # <--- NEW: Close DB connection
    
    if ngrok_tunnel:
        try:
            from pyngrok import ngrok 
            ngrok.disconnect(ngrok_tunnel.public_url)
        except Exception:
            pass
    print("Backend shutting down.")

app = FastAPI(lifespan=lifespan)

# --- 2) CLINICAL KNOWLEDGE BASE & IVR HELPERS ---
# ... (CLINICAL_QUESTIONS and helper functions remain unchanged) ...
CLINICAL_QUESTIONS = {
    "shortness of breath": ["Are you finding it hard to breathe when you are resting?", "Do you have to sit up to sleep because of your breathing?"],
    "swelling": ["Have your feet, ankles, or legs gotten bigger or more swollen?", "How much heavier do you feel in the last two days?"], 
    "cough": ["Are you coughing up any mucus or phlegm?", "What color is the mucus you are coughing up?"], 
    "dizziness": ["Did you feel very dizzy or faint after taking your blood pressure pills?", "Did you almost fall down or lose your balance?"],
    "furosemide": ["Did you take your water pill (Furosemide) today?", "Are you feeling very thirsty or having muscle pains after taking it?"],
    "lisinopril": ["Did you take your blood pressure pill (Lisinopril) today?", "Do you have a dry, constant cough?"],
    "salbutamol": ["Did you need to use your puff/inhaler (Salbutamol) more often today?", "Do you feel it is helping you breathe better?"],
}

def handle_session_error():
    """Generic error response and state cleanup for simplicity in IVR flow."""
    response = VoiceResponse()
    response.say("I apologize, but there was a system error. Goodbye.")
    response.hangup()
    return Response(content=str(response), media_type="application/xml")

# --- ML and Risk Helpers (Updated to work with dictionary output from DB) ---
# ... (generate_synthetic_data, train_ml_model, clinical_ner_extract, get_contextual_questions, perform_nlp, predict_and_explain remain UNCHANGED) ...
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

def clinical_ner_extract(text):
    text_lower = text.lower()
    extracted = set()
    if any(word in text_lower for word in ['cough', 'mucus', 'phlegm']): extracted.add("cough")
    if any(word in text_lower for word in ['breath', 'wheez', 'shortness of breath']): extracted.add("shortness of breath")
    if any(word in text_lower for word in ['swell', 'ankle', 'feet', 'leg', 'fluid', 'heavy']): extracted.add("swelling")
    if any(word in text_lower for word in ['dizzy', 'light-head', 'faint']): extracted.add("dizziness")
    return list(extracted)

def get_contextual_questions(patient_id, user_text):
    _, patient = get_patient_context(patient_id, by_phone=False) # Get patient data from DB
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
    if GLOBAL_MODEL is None or GLOBAL_EXPLAINER is None:
        return 0.0, [{"feature": "System Error", "impact_score": 0.0}]
        
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
    # Fetch latest data
    _, patient = get_patient_context(patient_id, by_phone=False)
    if not patient: return None

    sentiment_score_neg = perform_nlp(symptoms_text)
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

    patient['daily_reports_log'].insert(0, new_report) # Latest report at index 0
    
    # Update the patient data dictionary for the database call
    patient['adherence'] = adherence_int
    patient['symptom_report'] = symptoms_text
    patient['sentiment_score'] = sentiment_score_neg
    patient['ml_prediction'] = round(risk_prob, 3)
    patient['risk_level'] = risk_level
    patient['shap_explanation'] = shap_explanation
    patient['last_call'] = new_report['date']

    # CRUCIAL: Save the updated dictionary back to the database
    update_patient_fields(patient_id, patient)

    if risk_level == "High":
        print(f"\n🚨🚨 CRITICAL ALERT: Patient {patient['name']} (ID: {patient_id}) - RISK: {risk_level} ({risk_prob*100:.1f}%)")
    return patient

# --- 4) IVR Endpoints (Twilio Webhooks) ---

@app.post("/twilio")
async def twilio_webhook(request: Request, Caller: str = Form(None), CallSid: str = Form(None)):
    patient_id, patient = get_patient_context(Caller) # <--- DB CALL
    if not patient_id:
        response = VoiceResponse()
        response.say("Sorry, we cannot find your file. Please call your care provider.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
        
    CALL_STATES[CallSid] = {
        "id": patient_id, "symptoms": [], "current_symptom_index": 0, "symptoms_text_log": "",
        "adherence_text_log": "", "call_sid": CallSid
    }
    
    response = VoiceResponse()
    gather = response.gather(input='speech', action=f'{PUBLIC_URL}/twilio/process_symptoms', timeout=10)
    gather.say(f"Hello {patient['name']}. This is your daily check-up. Please tell us briefly how you are feeling today. Mention anything new, like being sick or feeling worse.")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_symptoms")
async def process_initial_symptoms(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return handle_session_error()
        
    symptoms_text = SpeechResult or "No symptoms reported"
    state['symptoms_text_log'] = symptoms_text
    patient_id = state['id']
    
    symptom_queue = get_contextual_questions(patient_id, symptoms_text)
    
    # ... (enqueue_symptoms definition omitted for brevity, assumed to be available)
    # Re-define enqueue_symptoms locally for completeness if it's not global
    def make_symptom_item(symptom_name):
        symptom_name = symptom_name.lower().strip()
        questions = CLINICAL_QUESTIONS.get(symptom_name, [f"Can you please tell us more about {symptom_name}?"])
        return {"name": symptom_name, "questions": questions, "q_index": 0, "answer_log": []}
        
    def enqueue_symptoms(state, symptoms_list):
        existing_names = {item["name"] for item in state.get("symptoms", [])}
        for symptom in symptoms_list:
            if symptom not in existing_names:
                state["symptoms"].append(make_symptom_item(symptom))
    
    enqueue_symptoms(state, symptom_queue)
    
    response = VoiceResponse()
    gather = response.gather(input='speech', action=f'{PUBLIC_URL}/twilio/process_adherence', timeout=5)
    gather.say("Thank you. Now, please tell us: did you take ALL of your pills and medicine today, exactly as the doctor told you? Please say 'Yes' or 'No'.")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_adherence")
async def process_adherence(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return handle_session_error()
    
    adherence_response = SpeechResult or "No speech response"
    state['adherence_text_log'] = adherence_response
    response = VoiceResponse()
    
    if not state.get("symptoms"):
        response.say("Thank you for telling us about your medicine. Since you reported nothing new, your file is stable. Goodbye.")
        perform_nlp_and_risk_update(state['id'], state['symptoms_text_log'], state['adherence_text_log']) # <--- DB UPDATE
        CALL_STATES.pop(CallSid, None)
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
        
    # Start the symptom follow-up queue
    current_symptom_item = state["symptoms"][state["current_symptom_index"]]
    current_q = current_symptom_item["questions"][current_symptom_item["q_index"]]
    
    gather = response.gather(input='speech', action=f'{PUBLIC_URL}/twilio/process_answer', timeout=10)
    gather.say(f"Okay, let's ask a few questions about your {current_symptom_item['name']}. {current_q}")
    return Response(content=str(response), media_type="application/xml")

@app.post("/twilio/process_answer")
async def process_answer(request: Request, SpeechResult: str = Form(None), Caller: str = Form(None), CallSid: str = Form(None)):
    state = CALL_STATES.get(CallSid)
    if not state: return handle_session_error()
    
    answer_text = SpeechResult or "No answer provided"
    response = VoiceResponse()
    current_idx = state.get("current_symptom_index", 0)
    
    if current_idx < len(state["symptoms"]):
        symptom_item = state["symptoms"][current_idx]
        current_q = symptom_item["questions"][symptom_item["q_index"]]
        symptom_item["answer_log"].append({"question": current_q, "answer": answer_text})
        symptom_item["q_index"] += 1
        
        # Next question for the same symptom
        if symptom_item["q_index"] < len(symptom_item["questions"]):
            next_q = symptom_item["questions"][symptom_item["q_index"]]
            gather = response.gather(input='speech', action=f'{PUBLIC_URL}/twilio/process_answer', timeout=10)
            gather.say(f"Next question about {symptom_item['name']}: {next_q}")
            return Response(content=str(response), media_type="application/xml")
            
        # Move to next symptom
        state["current_symptom_index"] += 1
        
    # Start asking questions for the next symptom in the queue
    if state["current_symptom_index"] < len(state["symptoms"]):
        next_item = state["symptoms"][state["current_symptom_index"]]
        next_q = next_item["questions"][next_item["q_index"]]
        gather = response.gather(input='speech', action=f'{PUBLIC_URL}/twilio/process_answer', timeout=10)
        gather.say(f"Now, about your {next_item['name']}. {next_q}")
        return Response(content=str(response), media_type="application/xml")
        
    # All questions processed. Finalize risk assessment.
    full_symptom_report = f"Initial: {state['symptoms_text_log']}"
    for item in state['symptoms']:
        for log in item['answer_log']:
            full_symptom_report += f" | {item['name']} Follow-up: {log['answer']}"
            
    updated_patient = perform_nlp_and_risk_update(state['id'], full_symptom_report, state['adherence_text_log']) # <--- DB UPDATE
    
    response.say("Thank you for answering all the questions. Your information is now updated.")
    if updated_patient and updated_patient['risk_level'] == "High":
        response.say("Your doctor has been immediately alerted to review your critical report.")
    response.say("Goodbye.")
    response.hangup()
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
        raise HTTPException(status_code=400, detail="Invalid Indian phone number format. Use E.164 starting with +91 followed by 10 digits.")
        
    # Check if patient exists in DB
    if get_patient_context(phone, by_phone=True)[0]:
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
    
    if not save_new_patient_record(patient_id, new_record): # <--- DB INSERT
        raise HTTPException(status_code=500, detail="Failed to save patient record to database.")

    print(f"\n✅ New Patient Added: {name} (ID: {patient_id})")
    return {"status": "success", "patient_id": patient_id, "name": name}

# --- 6) Dashboard and Intervention Endpoints (APIs for Streamlit) ---
@app.get("/api/patients/all_summary")
async def get_all_patients_summary():
    all_patients = get_all_patients_data() # <--- DB QUERY
    summary = []
    for p in all_patients:
        summary.append({
            "id": p['id'], "name": p['name'], "risk_level": p['risk_level'],
            "risk_probability": p['ml_prediction'] * 100, "last_report": p['symptom_report'],
            "shap_explanation": p['shap_explanation'], "doctor_override": p['doctor_override'],
            "intervention_notes": p['intervention_notes'], "last_call": p['last_call']
        })
    summary.sort(key=lambda p: p['risk_probability'], reverse=True)
    return JSONResponse(content=summary)

@app.get("/api/patients/{patient_id}/history")
async def get_patient_history_api(patient_id: str):
    patient_id_found, patient = get_patient_context(patient_id, by_phone=False) # <--- DB QUERY
    if not patient_id_found:
        raise HTTPException(status_code=404, detail="Patient not found")

    response_data = {
        "patient_info": {"id": patient_id, "name": patient['name'], "age": patient['age'], "phone": patient['phone']},
        "discharge_summary": patient['discharge_summary'],
        "daily_data_log": patient.get('daily_reports_log', []),
    }
    return JSONResponse(content=response_data)

@app.post("/api/patients/intervene/{patient_id}")
async def log_intervention_api(patient_id: str, notes: Request):
    patient_id_found, patient = get_patient_context(patient_id, by_phone=False) # <--- DB QUERY
    if not patient_id_found:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    notes_json = await notes.json()
    intervention_notes = notes_json.get("notes", "No notes provided")
    
    # Update the patient dictionary
    patient['doctor_override'] = True
    patient['intervention_notes'] = intervention_notes
    
    # Save the updated dictionary back to the database
    update_patient_fields(patient_id, patient) # <--- DB UPDATE

    print(f"\n👨‍⚕️ INTERVENTION LOGGED for {patient['name']}")
    return {"status": "success", "patient_id": patient_id, "message": "Intervention logged."}


# --- 7) Scheduling and Server Control ---
def call_patient(patient_id, phone_number):
    global PUBLIC_URL, twilio_client
    
    if twilio_client is None:
        print(f"Skipping call for {patient_id}: Twilio client is not available.")
        return
        
    try:
        # Use PUBLIC_URL dynamically for the webhook callback
        call = twilio_client.calls.create(
            to=phone_number, from_=TWILIO_NUMBER, url=f"{PUBLIC_URL}/twilio"
        )
        print(f"Call initiated for {patient_id}. SID: {call.sid}")
    except Exception as e:
        print(f"Error calling {patient_id}: {e}")

def call_patients_job():
    print(f"\n--- Running Daily Automated Call Job @ {datetime.now().strftime('%H:%M:%S')} ---")
    all_patients = get_all_patients_data() # <--- DB QUERY
    
    if not all_patients:
        print("No patients enrolled in the system.")
        return
        
    for patient in all_patients:
        if patient.get('risk_level', 'Low') == 'Low' and not patient.get('doctor_override', False):
            call_patient(patient['id'], patient['phone'])
        else:
            print(f"Skipping automated call for {patient['name']}. Status: Risk={patient.get('risk_level', 'Low')}, Override={patient.get('doctor_override', False)}")
            
    return {"status": "job_completed", "count": len(all_patients)}

@app.post("/call_patients_job_manual")
async def manual_call_trigger():
    result = call_patients_job()
    return JSONResponse(content=result)

@app.get("/get_public_url")
async def get_public_url_api():
    """Endpoint for Streamlit to fetch the FastAPI URL"""
    if not PUBLIC_URL or PUBLIC_URL == "http://localhost:8000":
         raise HTTPException(status_code=503, detail="Public URL is not ready. Check deployment or Ngrok status.")
    return {"public_url": PUBLIC_URL}
