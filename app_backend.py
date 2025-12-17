from fastapi import FastAPI, Form, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSON
from typing import List, Dict, Optional
from datetime import datetime
import os, re
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

import spacy
from textblob import TextBlob

from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# =====================================================
# LOAD NER MODEL (spaCy Medical)
# =====================================================

try:
    nlp = spacy.load("en_core_sci_sm")
except Exception as e:
    raise RuntimeError("spaCy model not loaded. Check requirements.txt")

# =====================================================
# DATABASE MODELS
# =====================================================

class Patient(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    phone: str = Field(index=True)
    age: int
    prior_admissions_30d: int
    comorbidity_score: int

    adherence: int = 1
    symptom_report: str = ""
    sentiment_score: float = 0.0
    ml_prediction: float = 0.0
    risk_level: str = "Low"
    last_call: Optional[str] = None

    doctor_override: bool = False
    intervention_notes: str = ""

    discharge_summary: Dict = Field(sa_column=Column(JSON))
    daily_reports_log: List[Dict] = Field(sa_column=Column(JSON, default=list))
    shap_explanation: List[Dict] = Field(sa_column=Column(JSON, default=list))

# =====================================================
# DATABASE
# =====================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

engine = create_engine(DATABASE_URL)

def get_session():
    with Session(engine) as s:
        yield s

# =====================================================
# TWILIO
# =====================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
CALL_STATE = {}

# =====================================================
# ML MODEL
# =====================================================

FEATURES = ["age", "prior_admissions_30d", "comorbidity_score", "adherence", "sentiment_score"]

def train_model():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(40, 90, 400),
        "prior_admissions_30d": np.random.randint(0, 4, 400),
        "comorbidity_score": np.random.randint(1, 10, 400),
        "adherence": np.random.randint(0, 2, 400),
        "sentiment_score": np.random.rand(400)
    })
    y = ((X.age > 70) | (X.adherence == 0)).astype(int)
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X, y)
    return model, shap.TreeExplainer(model)

MODEL, EXPLAINER = train_model()

# =====================================================
# NER + NLP PIPELINE
# =====================================================

def extract_medical_entities(text: str) -> Dict:
    doc = nlp(text.lower())

    symptoms = set()
    meds = set()

    for ent in doc.ents:
        if ent.label_ in {"DISEASE", "SYMPTOM"}:
            symptoms.add(ent.text)
        if ent.label_ in {"CHEMICAL", "DRUG"}:
            meds.add(ent.text)

    sentiment = TextBlob(text).sentiment.polarity

    return {
        "symptoms": list(symptoms),
        "medications": list(meds),
        "sentiment": sentiment
    }

# =====================================================
# RISK PREDICTION
# =====================================================

def compute_risk(patient: Patient):
    row = pd.DataFrame([{
        "age": patient.age,
        "prior_admissions_30d": patient.prior_admissions_30d,
        "comorbidity_score": patient.comorbidity_score,
        "adherence": patient.adherence,
        "sentiment_score": patient.sentiment_score
    }])
    prob = MODEL.predict_proba(row)[0][1]
    shap_vals = EXPLAINER.shap_values(row)[1]

    patient.ml_prediction = float(prob)
    patient.risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.5 else "Low"
    patient.shap_explanation = [
        {"feature": FEATURES[i], "impact": round(float(shap_vals[i]), 3)}
        for i in range(len(FEATURES))
    ]

# =====================================================
# FASTAPI APP
# =====================================================

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

# =====================================================
# API ENDPOINTS (MATCH FRONTEND)
# =====================================================

@app.post("/api/patients/add")
async def add_patient(
    session: Session = Depends(get_session),
    name: str = Form(...),
    phone: str = Form(...),
    age: int = Form(...),
    prior_admissions_30d: int = Form(...),
    comorbidity_score: int = Form(...),
    discharge_diagnosis: str = Form(...),
    medications: str = Form(...)
):
    phone = "+91" + re.sub(r"\D", "", phone)[-10:]

    pid = f"P{int(datetime.utcnow().timestamp())}"

    patient = Patient(
        id=pid,
        name=name,
        phone=phone,
        age=age,
        prior_admissions_30d=prior_admissions_30d,
        comorbidity_score=comorbidity_score,
        discharge_summary={
            "diagnosis": discharge_diagnosis,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "medications": medications
        }
    )
    session.add(patient)
    session.commit()
    return {"status": "success", "patient_id": pid}

@app.get("/api/patients/all_summary")
def all_summary(session: Session = Depends(get_session)):
    return [{
        "id": p.id,
        "name": p.name,
        "risk_level": p.risk_level,
        "risk_probability": p.ml_prediction,
        "last_report": p.symptom_report,
        "shap_explanation": p.shap_explanation,
        "doctor_override": p.doctor_override,
        "intervention_notes": p.intervention_notes,
        "last_call": p.last_call
    } for p in session.exec(select(Patient)).all()]

@app.get("/api/patients/{pid}/history")
def history(pid: str, session: Session = Depends(get_session)):
    p = session.get(Patient, pid)
    if not p:
        raise HTTPException(404)
    return {
        "patient_info": {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "phone": p.phone
        },
        "discharge_summary": p.discharge_summary,
        "daily_data_log": p.daily_reports_log
    }

# =====================================================
# TWILIO IVR WITH NER
# =====================================================

@app.post("/twilio")
async def twilio_entry(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")

    CALL_STATE[call_sid] = {}
    vr = VoiceResponse()
    vr.say("Hello. How are you feeling today?")
    vr.record(action="/twilio/process", method="POST", timeout=5)
    return Response(str(vr), media_type="application/xml")

@app.post("/twilio/process")
async def process_speech(request: Request, session: Session = Depends(get_session)):
    form = await request.form()
    speech = form.get("SpeechResult", "")
    call_sid = form.get("CallSid")

    entities = extract_medical_entities(speech)

    patient = session.exec(select(Patient).where(Patient.phone == form.get("To"))).first()
    if not patient:
        return Response("<Response><Say>Thank you.</Say></Response>", media_type="application/xml")

    patient.symptom_report = speech
    patient.sentiment_score = entities["sentiment"]
    patient.adherence = 0 if "miss" in speech.lower() else 1

    compute_risk(patient)

    patient.daily_reports_log.append({
        "date": datetime.utcnow().isoformat(),
        "symptoms": entities["symptoms"],
        "medications": entities["medications"],
        "sentiment": patient.sentiment_score,
        "risk": patient.risk_level
    })

    session.add(patient)
    session.commit()

    vr = VoiceResponse()
    if patient.risk_level == "High":
        vr.say("Thank you. A doctor will contact you shortly.")
    else:
        vr.say("Thank you. Please continue your medications.")
    return Response(str(vr), media_type="application/xml")

# =====================================================
# MANUAL CALL TRIGGER
# =====================================================

@app.post("/call_patients_job_manual")
def manual_call(session: Session = Depends(get_session)):
    count = 0
    for p in session.exec(select(Patient)).all():
        if not p.doctor_override:
            twilio_client.calls.create(
                to=p.phone,
                from_=TWILIO_PHONE,
                url=f"{PUBLIC_URL}/twilio"
            )
            count += 1
    return {"count": count}
