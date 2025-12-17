import streamlit as st
import requests
import pandas as pd
import re

FASTAPI_BASE_URL = "https://ivr-clinical-backend.onrender.com"

def fetch(endpoint):
    r = requests.get(f"{FASTAPI_BASE_URL}{endpoint}")
    r.raise_for_status()
    return r.json()

def post(endpoint, data):
    r = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", data=data)
    r.raise_for_status()
    return r.json()

def normalize(phone):
    phone = re.sub(r"[^\d]", "", phone)
    return f"+91{phone}"

st.set_page_config(page_title="IVR Dashboard", layout="wide")

tab1, tab2 = st.tabs(["📊 Dashboard", "➕ Enroll Patient"])

with tab2:
    st.header("Enroll Patient")
    with st.form("enroll"):
        name = st.text_input("Name")
        phone = st.text_input("Phone (10 digits)")
        age = st.number_input("Age", 18, 120, 60)
        adm = st.number_input("Prior Admissions", 0, 5, 0)
        com = st.slider("Comorbidity Score", 1, 7, 3)
        diag = st.text_area("Diagnosis")
        meds = st.text_area("Medications")
        if st.form_submit_button("Enroll"):
            data = {
                "name": name,
                "phone": normalize(phone),
                "age": age,
                "prior_admissions_30d": adm,
                "comorbidity_score": com,
                "discharge_diagnosis": diag,
                "medications": meds,
            }
            res = post("/api/patients/add", data)
            st.success(f"Patient added: {res['patient_id']}")

with tab1:
    st.header("Patient Dashboard")
    data = fetch("/api/patients/all_summary")
    if data:
        df = pd.DataFrame(data)
        df["risk_probability"] = (df["risk_probability"] * 100).round(1)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No patients enrolled.")
