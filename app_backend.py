# -*- coding: utf-8 -*-
"""app_backend.py: Production-Ready FastAPI Backend with Persistent PostgreSQL Database"""

# --- 0) Setup & Imports ---
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import uvicorn, time, os, json, pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler 
from contextlib import asynccontextmanager 
from datetime import datetime, date
import requests
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from typing import Optional, Any

# --- DATABASE IMPORTS & MODELS ---
from sqlmodel import Field, SQLModel, create_engine, Session, select 
from sqlalchemy import Column # <--- ADDED: For correct JSON column definition
from sqlalchemy.dialects.postgresql import JSON as PG_JSON # <--- ADDED: Use PostgreSQL's JSON type
from pydantic import BaseModel 

# Pydantic Models for Data Structure (for SQLModel JSON columns)
class DailyReport(SQLModel):
    date: str
    adherence: int
    symptom_report: str
    sentiment_score: float
    ml_prediction: float
    risk_level: str
    shap_explanation: list[dict] 

class DischargeSummary(SQLModel):
    diagnosis: str
    date: str
    medications: str
    notes: str

# SQLModel for the Patient Table
class Patient(SQLModel, table=True):
    id: str = Field(primary_key=True)
    phone: str = Field(index=True)
    name: str
    age: int
    prior_admissions_30d: int
    comorbidity_score: int
    
    # Dynamic/Latest Status
    adherence: int = 1
    symptom_report: str = "New Patient Enrolled."
    sentiment_score: float = 0.0
    ml_prediction: float = 0.0
    risk_level: str = "Low"
    last_call: str | None = None
    doctor_override: bool = False
    intervention_notes: str = ""
    
    # Static & Log Data (Stored as JSON)
    # CORRECTED JSON FIELD DEFINITIONS (Fixes TypeError in Python 3.13)
    discharge_summary: DischargeSummary = Field(
        sa_column=Column(PG_JSON, default=DischargeSummary(diagnosis="", date="", medications="", notes="").model_dump())
    )
    daily_reports_log: list[DailyReport] = Field(
        sa_column=Column(PG_JSON, default=[])
    )
    # The previous problematic field:
    shap_explanation: list[dict] = Field(
        sa_column=Column(PG_JSON, default=[])
    )


# Pydantic Model for PATCH/UPDATE operations (only update specified fields)
class PatientUpdate(BaseModel):
    phone: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    prior_admissions_30d: Optional[int] = None
    comorbidity_score: Optional[int] = None
    doctor_override: Optional[bool] = None
    intervention_notes: Optional[str] = None

# --- 1) Configuration & Initialization ---

# Get DB URL from Render Environment Variables
DATABASE_URL = os.environ.get("DATABASE_URL") 
if not DATABASE_URL:
    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    DATABASE_URL = sqlite_url
    print(f"WARNING: Using SQLite at {sqlite_url}. Deploy with Render PostgreSQL for persistence.")

# Modify URL for asyncpg compatibility if needed (Render often requires this)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) 

# --- DB Helper Functions ---

def create_db_and_tables():
    """Initializes the database and creates the tables based on the models."""
    SQLModel.metadata.create_all(engine)
    
def get_session():
    """Dependency to provide a database session."""
    with Session(engine) as session:
        yield session

# --- Twilio and Scheduler Configuration (continued) ---

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
DAILY_CALL_HOUR_IST = 10

# Initialize client only if credentials are available
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    print("WARNING: Twilio credentials not found. Calls will not be placed.")
    twilio_client = None

scheduler = AsyncIOScheduler() 
GLOBAL_MODEL, GLOBAL_EXPLAINER = None, None
PUBLIC_URL = "" 
PATIENT_ID_COUNTER = 1001 
CALL_STATES = {} 

# Helper Functions 

def get_next_patient_id(session: Session):
    """Generates the next patient ID based on the highest existing ID in the DB."""
    try:
        highest_id_obj = session.exec(
            select(Patient.id).order_by(Patient.id.desc())
        ).first()
        if highest_id_obj and highest_id_obj.startswith('P'):
            global PATIENT_ID_COUNTER
            PATIENT_ID_COUNTER = int(highest_id_obj[1:]) + 1
    except Exception:
        pass 
    new_id = f"P{PATIENT_ID_COUNTER}"
    PATIENT_ID_COUNTER += 1
    return new_id

def normalize_phone_number(phone):
    """Normalizes a phone number string to E.164 format (+91XXXXXXXXXX) for Indian numbers."""
    clean_phone = re.sub(r'[^\d+]', '', phone)
    if clean_phone.startswith('+91') and len(clean_phone) == 13: return clean_phone
    if clean_phone.startswith('91') and len(clean_phone) == 12: return f"+{clean_phone}"
    if len(clean_phone) == 10: return f"+91{clean_phone}"
    return clean_phone 

def get_patient_context(phone_number, session: Session):
    """Looks up patient by phone number using the database session."""
    normalized_caller = normalize_phone_number(phone_number)
    
    statement = select(Patient).where(Patient.phone == normalized_caller)
    patient = session.exec(statement).first()
    
    if patient:
        return patient.id, patient.model_dump() 
        
    return None, None

def perform_nlp(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['worse', 'pain', 'barely breathe', 'critical']):
        return 0.90
    elif any(word in text_lower for word in ['fine', 'better', 'good', 'stable']):
        return 0.10
    else:
        return 0.50

def predict_and_explain(patient_features):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER
    
    # Check if model is loaded (should be loaded in lifespan, but safe check here)
    if not GLOBAL_MODEL or not GLOBAL_EXPLAINER:
        print("WARNING: ML Model not loaded. Using default risk.")
        return 0.5, [{"feature": "System Status", "impact_score": 0.5}]

    features = {col: [patient_features.get(col)] for col in FEATURE_COLUMNS}
    X_single = pd.DataFrame(features)
    readmission_prob = GLOBAL_MODEL.predict_proba(X_single)[:, 1][0]
    shap_values = GLOBAL_EXPLAINER.shap_values(X_single)[1]
    explanation = sorted(zip(FEATURE_COLUMNS, shap_values), key=lambda x: abs(x[1]), reverse=True)
    formatted_explanation = [
        {"feature": name, "impact_score": round(score, 3)}
        for name, score in explanation[:5]
    ]
    return readmission_prob, formatted_explanation

def perform_nlp_and_risk_update(patient_id, symptoms_text, adherence_text, session: Session):
    patient = session.get(Patient, patient_id)
    if not patient: return None

    # Use the model_dump() for features dictionary
    patient_features = patient.model_dump(include=set(FEATURE_COLUMNS))
    
    sentiment_score_neg = perform_nlp(symptoms_text)
    # Check for simple 'yes' or 'no' response for adherence
    adherence_int = 1 if 'yes' in adherence_text.lower() and 'not' not in adherence_text.lower() else 0
    
    patient_features['adherence'] = adherence_int
    patient_features['sentiment_score_neg'] = sentiment_score_neg

    risk_prob, shap_explanation = predict_and_explain(patient_features)

    if risk_prob >= 0.70: risk_level = "High"
    elif risk_prob >= 0.50: risk_level = "Medium"
    else: risk_level = "Low"

    # Create the new DailyReport object
    new_report = DailyReport(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        adherence=adherence_int,
        symptom_report=symptoms_text,
        sentiment_score=sentiment_score_neg,
        ml_prediction=round(risk_prob, 3),
        risk_level=risk_level,
        shap_explanation=shap_explanation,
    )

    # Update patient object fields
    # Ensure daily_reports_log is treated as a list of dicts for the JSON column
    daily_log = patient.daily_reports_log or []
    daily_log.insert(0, new_report.model_dump())
    patient.daily_reports_log = daily_log
    
    patient.adherence = adherence_int
    patient.symptom_report = symptoms_text
    patient.sentiment_score = sentiment_score_neg
    patient.ml_prediction = round(risk_prob, 3)
    patient.risk_level = risk_level
    patient.shap_explanation = shap_explanation
    patient.last_call = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    session.add(patient)
    session.commit()
    session.refresh(patient) 

    if risk_level == "High":
        print(f"\n🚨🚨 CRITICAL ALERT: Patient {patient.name} (ID: {patient_id}) - RISK: {risk_level} ({risk_prob*100:.1f}%)")
    
    return patient.model_dump() 

# --- DUMMY ML/CLINICAL HELPERS (Unchanged Logic) ---

FEATURE_COLUMNS = ['age', 'prior_admissions_30d', 'comorbidity_score', 'adherence', 'sentiment_score_neg']
def generate_synthetic_data(num_samples=200):
    np.random.seed(42)
    data = {
        'age': np.random.randint(40, 95, num_samples), 'prior_admissions_30d': np.random.randint(0, 4, num_samples), 
        'comorbidity_score': np.random.randint(1, 8, num_samples), 'adherence': np.random.randint(0, 2, num_samples), 
        'sentiment_score_neg': np.random.uniform(0.1, 0.9, num_samples)
    }
    df = pd.DataFrame(data)
    risk = ((df['age'] > 70) * 0.3 + (df['prior_admissions_30d'] > 1) * 0.4 + (df['comorbidity_score'] > 4) * 0.2 + (df['adherence'] == 0) * 0.5 + (df['sentiment_score_neg'] > 0.6) * 0.3)
    prob = np.clip(risk / risk.max() + np.random.uniform(-0.1, -0.1, num_samples), 0, 1) # Note: Used negative uniform for synthetic realism
    df['readmission_status'] = (prob > 0.5).astype(int)
    return df

def train_ml_model():
    data = generate_synthetic_data()
    X = data[FEATURE_COLUMNS]
    y = data['readmission_status']
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', learning_rate=0.1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer

CLINICAL_QUESTIONS = {
    "shortness of breath": ["Are you finding it hard to breathe when you are resting?", "Do you have to sit up to sleep because of your breathing?"],
    "swelling": ["Have your feet, ankles, or legs gotten bigger or more swollen?", "How much heavier do you feel in the last two days?"], 
    "cough": ["Are you coughing up any mucus or phlegm?", "What color is the mucus you are coughing up?"], 
    "dizziness": ["Did you feel very dizzy or faint after taking your blood pressure pills?", "Did you almost fall down or lose your balance?"],
    "furosemide": ["Did you take your water pill (Furosemide) today?", "Are you feeling very thirsty or having muscle pains after taking it?"],
    "lisinopril": ["Did you take your blood pressure pill (Lisinopril) today?", "Do you have a dry, constant cough?"],
    "salbutamol": ["Did you need to use your puff/inhaler (Salbutamol) more often today?", "Do you feel it is helping you breathe better?"],
}

def clinical_ner_extract(text):
    text_lower = text.lower()
    extracted = set()
    if any(word in text_lower for word in ['cough', 'mucus', 'phlegm', 'sputum']): extracted.add("cough")
    if any(word in text_lower for word in ['breath', 'wheez', 'gaspin', 'pant', 'gasp', 'shortness of breath']): extracted.add("shortness of breath")
    if any(word in text_lower for word in ['swell', 'ankle', 'feet', 'leg', 'fluid', 'heavy', 'tight shoes']): extracted.add("swelling")
    if any(word in text_lower for word in ['dizzy', 'light-head', 'faint', 'unstable', 'almost fall']): extracted.add("dizziness")
    return list(extracted)

def get_contextual_questions(patient_data, user_text):
    explicit_symptoms = set(clinical_ner_extract(user_text))
    implied_symptoms = set()
    diagnosis = patient_data['discharge_summary']['diagnosis'].lower()
    medications = patient_data['discharge_summary']['medications'].lower()
    if 'chf' in diagnosis or 'heart failure' in diagnosis: implied_symptoms.update(["swelling", "shortness of breath"])
    if 'copd' in diagnosis or 'bronchitis' in diagnosis: implied_symptoms.update(["cough", "shortness of breath"])
    for key in CLINICAL_QUESTIONS.keys():
        if key.lower() in medications.lower(): implied_symptoms.add(key)
    final_queue = list(explicit_symptoms)
    for s in implied_symptoms:
        if s not in explicit_symptoms: final_queue.append(s)
    return final_queue

def make_symptom_item(symptom_name):
    symptom_name = symptom_name.lower().strip()
    questions = CLINICAL_QUESTIONS.get(symptom_name, [f"Can you please tell us more about {symptom_name}?"])
    return {"name": symptom_name, "questions": questions, "q_index": 0, "answer_log": []}

def enqueue_symptoms(state, symptoms_list):
    existing_names = {item["name"] for item in state.get("symptoms", [])}
    for symptom in symptoms_list:
        if symptom not in existing_names:
            state["symptoms"].append(make_symptom_item(symptom))

# --- Lifespan Manager & App Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER, PUBLIC_URL
    
    create_db_and_tables() 
    
    PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")

    print("\n" + "="*70)
    print(f"FastAPI Backend Starting. Public URL for Twilio: {PUBLIC_URL}")

    GLOBAL_MODEL, GLOBAL_EXPLAINER = train_ml_model()

    hour = DAILY_CALL_HOUR_IST
    scheduler.remove_all_jobs()
    scheduler.add_job(call_patients_job_scheduled, trigger="cron", hour=hour, minute=0, timezone=pytz.timezone('Asia/Kolkata'))
    scheduler.start()
    print(f"Daily Scheduler set for {DAILY_CALL_HOUR_IST}:00 IST.")
    print("="*70 + "\n")

    yield 

    scheduler.shutdown()
    print("Backend shutting down.")

app = FastAPI(lifespan=lifespan)

# --- 4) IVR Endpoints (Twilio Webhooks) ---

@app.post("/twilio")
async def twilio_webhook(request: Request, Caller: str = Form(None), CallSid: str = Form(None)):
    with Session(engine) as session: 
        patient_id, patient = get_patient_context(Caller, session)
        
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
    
    with Session(engine) as session: 
        _, patient_data = get_patient_context(Caller, session)
        symptom_queue = get_contextual_questions(patient_data, symptoms_text)
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
        with Session(engine) as session: 
            perform_nlp_and_risk_update(state['id'], state['symptoms_text_log'], state['adherence_text_log'], session)
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
            
    with Session(engine) as session: 
        updated_patient = perform_nlp_and_risk_update(
            state['id'], full_symptom_report, state['adherence_text_log'], session
        )
    
    response.say("Thank you for answering all the questions. Your information is now updated.")
    if updated_patient and updated_patient['risk_level'] == "High":
        response.say("Your doctor has been immediately alerted to review your report.")
    response.say("Goodbye.")
    CALL_STATES.pop(CallSid, None)
    return Response(content=str(response), media_type="application/xml")

# --- 5) Data Management Endpoints (CRUD) ---

@app.post("/api/patients/add")
async def add_new_patient_api(
    session: Session = Depends(get_session),
    name: str = Form(...), phone: str = Form(...), age: int = Form(...),
    prior_admissions_30d: int = Form(...), comorbidity_score: int = Form(...),
    discharge_diagnosis: str = Form(...), medications: str = Form(...) 
):
    if not phone.startswith('+91') or not re.match(r'^\+91\d{10}$', phone):
        raise HTTPException(status_code=400, detail="Invalid Indian phone number format. Use E.164 starting with +91.")
    
    normalized_phone = normalize_phone_number(phone)
    
    if session.exec(select(Patient).where(Patient.phone == normalized_phone)).first():
        raise HTTPException(status_code=400, detail=f"Patient with phone {phone} already exists.")
        
    patient_id = get_next_patient_id(session)
    
    new_patient = Patient(
        id=patient_id, phone=normalized_phone, name=name, age=age, prior_admissions_30d=prior_admissions_30d, 
        comorbidity_score=comorbidity_score, 
        discharge_summary=DischargeSummary(
            diagnosis=discharge_diagnosis, date=datetime.now().strftime("%Y-%m-%d"),
            medications=medications, notes="Patient enrolled post-discharge for 30-day monitoring."
        )
    )
    
    session.add(new_patient)
    session.commit()
    session.refresh(new_patient) 
    
    print(f"\n✅ New Patient Added: {name} (ID: {patient_id})")
    return {"status": "success", "patient_id": patient_id, "name": name, "phone": new_patient.phone}

@app.patch("/api/patients/update/{patient_id}")
async def update_patient_api(
    patient_id: str, 
    patient_data: PatientUpdate, 
    session: Session = Depends(get_session)
):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    update_data = patient_data.model_dump(exclude_unset=True)
    
    if 'phone' in update_data:
        new_phone = normalize_phone_number(update_data['phone'])
        if new_phone != normalize_phone_number(patient.phone):
            if session.exec(select(Patient).where(Patient.phone == new_phone)).first():
                raise HTTPException(status_code=400, detail=f"Phone number {new_phone} is already registered to another patient.")
        patient.phone = new_phone
        update_data.pop('phone') 

    for key, value in update_data.items():
        setattr(patient, key, value)
        
    session.add(patient)
    session.commit()
    session.refresh(patient)
    
    print(f"\n📝 Patient Data Updated: {patient.name} (ID: {patient_id})")
    return patient.model_dump()

@app.delete("/api/patients/delete/{patient_id}")
async def delete_patient_api(patient_id: str, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    session.delete(patient)
    session.commit()
    
    return {"status": "success", "message": f"Patient {patient_id} deleted."}


# --- 6) Dashboard and Intervention Endpoints (Unchanged) ---

@app.get("/api/patients/all_summary")
async def get_all_patients_summary(session: Session = Depends(get_session)):
    patients = session.exec(select(Patient)).all()
    summary = []
    for p in patients:
        p_dict = p.model_dump()
        summary.append({
            "id": p_dict['id'], "name": p_dict['name'], "risk_level": p_dict['risk_level'],
            "risk_probability": p_dict['ml_prediction'] * 100, "last_report": p_dict['symptom_report'],
            "shap_explanation": p_dict['shap_explanation'], "doctor_override": p_dict['doctor_override'],
            "intervention_notes": p_dict['intervention_notes'], "last_call": p_dict['last_call']
        })
    summary.sort(key=lambda p: p['risk_probability'], reverse=True)
    return JSONResponse(content=summary)

@app.get("/api/patients/{patient_id}/history")
async def get_patient_history_api(patient_id: str, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    p_dict = patient.model_dump()
    response_data = {
        "patient_info": {"id": patient_id, "name": p_dict['name'], "age": p_dict['age'], "phone": p_dict['phone']},
        "discharge_summary": p_dict['discharge_summary'],
        "daily_data_log": p_dict['daily_reports_log'],
    }
    return JSONResponse(content=response_data)

@app.post("/api/patients/intervene/{patient_id}")
async def log_intervention_api(patient_id: str, notes: Request, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    notes_json = await notes.json()
    intervention_notes = notes_json.get("notes", "No notes provided")
    
    patient.doctor_override = True
    patient.intervention_notes = intervention_notes
    
    session.add(patient)
    session.commit()
    
    print(f"\n👨‍⚕️ INTERVENTION LOGGED for {patient.name}")
    return {"status": "success", "patient_id": patient_id, "message": "Intervention logged."}


# --- 7) Scheduling and Server Control (Unchanged) ---

def call_patient(patient_id, phone_number):
    global PUBLIC_URL, twilio_client
    if not twilio_client:
        print(f"Cannot call {patient_id}. Twilio client is not initialized.")
        return
        
    try:
        call = twilio_client.calls.create(
            to=phone_number, from_=TWILIO_NUMBER, url=f"{PUBLIC_URL}/twilio"
        )
        print(f"Call initiated for {patient_id}. SID: {call.sid}")
    except Exception as e:
        print(f"Error calling {patient_id}: {e}")

# Scheduled job function
def call_patients_job_scheduled():
    with Session(engine) as session:
        print(f"\n--- Running SCHEDULED Call Job @ {datetime.now().strftime('%H:%M:%S')} ---")
        patients = session.exec(select(Patient)).all()
        if not patients:
            print("No patients enrolled in the system.")
            return

        for patient in patients:
            # Only call if risk is not high and no doctor override is active
            if patient.risk_level != 'High' and not patient.doctor_override:
                call_patient(patient.id, patient.phone)
            else:
                print(f"Skipping automated call for {patient.name}.")
        print("--- Scheduled Call Job Finished ---")

# Manual job trigger endpoint
@app.post("/call_patients_job_manual")
async def manual_call_trigger(session: Session = Depends(get_session)):
    print(f"\n--- Running MANUAL Call Job @ {datetime.now().strftime('%H:%M:%S')} ---")
    patients = session.exec(select(Patient)).all()
    if not patients:
        print("No patients enrolled in the system.")
        return {"status": "job_completed", "count": 0}

    count = 0
    for patient in patients:
        if patient.risk_level != 'High' and not patient.doctor_override:
            call_patient(patient.id, patient.phone)
            count += 1
        else:
            print(f"Skipping automated call for {patient.name}.")
            
    return {"status": "job_completed", "count": count, "message": f"{count} calls initiated."}

@app.get("/get_public_url")
async def get_public_url_api():
    """Endpoint for Streamlit to fetch the FastAPI URL"""
    return {"public_url": PUBLIC_URL}
