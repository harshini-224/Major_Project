# -*- coding: utf-8 -*-
"""
Production-Ready FastAPI Backend
Safe for Render + PostgreSQL + Streamlit
"""

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from datetime import datetime
import os, re, pytz

import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from sqlmodel import SQLModel, Field, Session, select, create_engine
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSON as PG_JSON
from pydantic import BaseModel
from typing import Optional

# ---------------- CONFIG ----------------

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///database.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

PUBLIC_URL = ""
scheduler = AsyncIOScheduler()
CALL_STATES = {}
PATIENT_ID_COUNTER = 1001

FEATURE_COLUMNS = [
    "age",
    "prior_admissions_30d",
    "comorbidity_score",
    "adherence",
    "sentiment_score_neg",
]

GLOBAL_MODEL = None
GLOBAL_EXPLAINER = None

# ---------------- MODELS ----------------

class Patient(SQLModel, table=True):
    id: str = Field(primary_key=True)
    phone: str = Field(index=True)
    name: str
    age: int
    prior_admissions_30d: int
    comorbidity_score: int

    adherence: int = 1
    symptom_report: str = "New Patient Enrolled."
    sentiment_score: float = 0.0
    ml_prediction: float = 0.0
    risk_level: str = "Low"
    last_call: Optional[str] = None
    doctor_override: bool = False
    intervention_notes: str = ""

    discharge_summary: dict = Field(
        sa_column=Column(PG_JSON),
        default_factory=lambda: {
            "diagnosis": "",
            "date": "",
            "medications": "",
            "notes": "",
        },
    )

    daily_reports_log: list = Field(
        sa_column=Column(PG_JSON),
        default_factory=list,
    )

    shap_explanation: list = Field(
        sa_column=Column(PG_JSON),
        default_factory=list,
    )


class PatientUpdate(BaseModel):
    phone: Optional[str] = None
    doctor_override: Optional[bool] = None
    intervention_notes: Optional[str] = None


# ---------------- HELPERS ----------------

def create_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


def normalize_phone(phone: str) -> str:
    phone = re.sub(r"[^\d]", "", phone)
    if len(phone) == 10:
        return f"+91{phone}"
    if phone.startswith("91") and len(phone) == 12:
        return f"+{phone}"
    if phone.startswith("+91") and len(phone) == 13:
        return phone
    raise ValueError("Invalid phone number")


def perform_nlp(text: str) -> float:
    text = text.lower()
    if any(w in text for w in ["worse", "pain", "breath", "critical"]):
        return 0.9
    if any(w in text for w in ["good", "better", "fine"]):
        return 0.1
    return 0.5


def generate_data(n=300):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(40, 90, n),
        "prior_admissions_30d": np.random.randint(0, 4, n),
        "comorbidity_score": np.random.randint(1, 7, n),
        "adherence": np.random.randint(0, 2, n),
        "sentiment_score_neg": np.random.rand(n),
    })
    risk = (
        (df.age > 70) * 0.3 +
        (df.prior_admissions_30d > 1) * 0.4 +
        (df.comorbidity_score > 4) * 0.3 +
        (df.adherence == 0) * 0.5 +
        (df.sentiment_score_neg > 0.6) * 0.3
    )
    df["readmit"] = (risk > 0.6).astype(int)
    return df


def train_model():
    data = generate_data()
    X = data[FEATURE_COLUMNS]
    y = data["readmit"]
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer


def predict(patient):
    X = pd.DataFrame([{k: patient[k] for k in FEATURE_COLUMNS}])
    prob = GLOBAL_MODEL.predict_proba(X)[0][1]
    shap_vals = GLOBAL_EXPLAINER.shap_values(X)[1]
    explanation = sorted(
        zip(FEATURE_COLUMNS, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return prob, [{"feature": f, "impact": round(v, 3)} for f, v in explanation[:5]]


# ---------------- APP LIFESPAN ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER, PUBLIC_URL
    create_db()
    GLOBAL_MODEL, GLOBAL_EXPLAINER = train_model()
    PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

# ---------------- API ----------------

@app.post("/api/patients/add")
def add_patient(
    name: str = Form(...),
    phone: str = Form(...),
    age: int = Form(...),
    prior_admissions_30d: int = Form(...),
    comorbidity_score: int = Form(...),
    discharge_diagnosis: str = Form(...),
    medications: str = Form(...),
    session: Session = Depends(get_session),
):
    phone = normalize_phone(phone)
    if session.exec(select(Patient).where(Patient.phone == phone)).first():
        raise HTTPException(400, "Patient already exists")

    global PATIENT_ID_COUNTER
    pid = f"P{PATIENT_ID_COUNTER}"
    PATIENT_ID_COUNTER += 1

    patient = Patient(
        id=pid,
        phone=phone,
        name=name,
        age=age,
        prior_admissions_30d=prior_admissions_30d,
        comorbidity_score=comorbidity_score,
        discharge_summary={
            "diagnosis": discharge_diagnosis,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "medications": medications,
            "notes": "Enrolled",
        },
    )

    session.add(patient)
    session.commit()

    return {"status": "success", "patient_id": pid}


@app.get("/api/patients/all_summary")
def summary(session: Session = Depends(get_session)):
    pts = session.exec(select(Patient)).all()
    return [{
        "id": p.id,
        "name": p.name,
        "risk_level": p.risk_level,
        "risk_probability": p.ml_prediction,
        "symptom_report": p.symptom_report,
        "doctor_override": p.doctor_override,
        "intervention_notes": p.intervention_notes,
        "last_call": p.last_call,
        "shap_explanation": p.shap_explanation,
    } for p in pts]


@app.get("/api/patients/{pid}/history")
def history(pid: str, session: Session = Depends(get_session)):
    p = session.get(Patient, pid)
    if not p:
        raise HTTPException(404)
    return {
        "patient_info": {"id": p.id, "name": p.name, "age": p.age, "phone": p.phone},
        "discharge_summary": p.discharge_summary,
        "daily_data_log": p.daily_reports_log,
    }


@app.post("/api/patients/intervene/{pid}")
def intervene(pid: str, payload: dict, session: Session = Depends(get_session)):
    p = session.get(Patient, pid)
    p.doctor_override = True
    p.intervention_notes = payload.get("notes", "")
    session.commit()
    return {"status": "ok"}
