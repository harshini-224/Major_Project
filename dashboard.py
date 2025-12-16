import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time
import re
import os

# --- Configuration: RENDER URL ---
# CRITICAL: Replace this placeholder with your actual Render service URL 
# (e.g., https://my-ivr-backend.onrender.com)
FASTAPI_BASE_URL = "https://YOUR_FASTAPI_RENDER_URL.onrender.com" 

# --- Helper Functions ---

def safe_date_format(date_str):
    """Safely converts an ISO date string (potentially with None/NaN) to a display format."""
    if not date_str or pd.isna(date_str) or str(date_str).lower() in ['never', 'none', 'nan']:
        return 'Never'
    
    if isinstance(date_str, str) and 'Z' in date_str:
        date_str = date_str.replace('Z', '+00:00')

    try:
        return datetime.fromisoformat(date_str).strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d %H:%M')
        except:
            return 'Invalid Date'

@st.cache_data(ttl=5) 
def fetch_patient_summary():
    """Fetches the main patient summary data from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/api/patients/all_summary")
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to the FastAPI backend at {FASTAPI_BASE_URL}. Please ensure the URL is correct and the service is running. Error: {e}")
        return None

def fetch_patient_history(patient_id):
    """Fetches detailed historical data for a single patient."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/api/patients/{patient_id}/history")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching history for {patient_id}. Error: {e}")
        return None

def log_intervention(patient_id, notes):
    """Posts a doctor's intervention note to the backend."""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/api/patients/intervene/{patient_id}", 
                                 data=json.dumps({"notes": notes}), 
                                 headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error logging intervention: {e}")
        return None

# --- UI Components (Same as previous version) ---

def render_alert_monitor(patient_data):
    st.header("🚨 Patient Tracking & Alert Monitor")

    if patient_data is None:
        st.stop()
    
    if not patient_data:
        st.info("No patients enrolled in the system yet. Use 'Enroll New Patient' to begin.")
        return

    df = pd.DataFrame(patient_data)
    
    df['Risk (%)'] = df['risk_probability'].apply(lambda x: f"{x:.1f}%")
    df['Last Call'] = df['last_call'].apply(safe_date_format)
    
    risk_order = {"High": 3, "Medium": 2, "Low": 1}
    df['risk_rank'] = df['risk_level'].apply(lambda x: risk_order.get(x, 0))
    df = df.sort_values(by=['risk_rank', 'risk_probability'], ascending=[False, False]).drop(columns=['risk_rank'])
    
    st.markdown("### 📋 Current Patient Status")
    st.dataframe(
        df[['id', 'name', 'risk_level', 'Risk (%)', 'last_report', 'Last Call', 'doctor_override']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "ID",
            "name": "Patient Name",
            "risk_level": "Risk Level",
            "last_report": "Latest Symptom Report",
            "doctor_override": st.column_config.CheckboxColumn("Intervention Logged", default=False)
        }
    )

    st.markdown("### 🔍 Patient Details and Intervention")
    selected_id = st.selectbox("Select Patient ID for Details:", df['id'].tolist() if not df.empty else [])

    if selected_id:
        patient_summary = df[df['id'] == selected_id].iloc[0]
        history = fetch_patient_history(selected_id)
        
        st.markdown(f"#### Details for {patient_summary['name']} ({selected_id})")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_color = "red" if patient_summary['risk_level'] == 'High' else "orange" if patient_summary['risk_level'] == 'Medium' else "green"
            st.metric(label="Current Risk Level", value=f":{risk_color}[{patient_summary['risk_level']}]")
            st.metric(label="Readmission Probability", value=patient_summary['Risk (%)'])
            st.markdown(f"**Last Report:** *{patient_summary['last_report']}*")

        with col2:
            st.markdown("**ML Risk Drivers (SHAP Explanation)**")
            if patient_summary['shap_explanation']:
                shap_df = pd.DataFrame(patient_summary['shap_explanation'])
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
            else:
                st.info("No ML explanation available yet.")
            
            st.markdown("---")
            st.markdown("#### Doctor Intervention Log")
            
            if patient_summary['doctor_override']:
                st.success(f"Intervention previously logged. Notes: {patient_summary['intervention_notes']}")
            
            with st.form(f"intervention_form_{selected_id}"):
                intervention_notes = st.text_area("Log Intervention/Follow-up Notes:", height=100)
                submitted = st.form_submit_button(f"Log Intervention for {selected_id}", type="primary")

                if submitted:
                    if intervention_notes:
                        log_intervention(selected_id, intervention_notes)
                        st.toast("Intervention logged successfully!", icon='✅')
                        st.rerun()
                    else:
                        st.warning("Please enter intervention notes.")

        st.markdown("---")
        st.markdown("### 📈 Historical Daily Reports")
        if history and history.get('daily_data_log'):
            history_df = pd.DataFrame(history['daily_data_log'])
            history_df['date'] = history_df['date'].apply(safe_date_format)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No historical reports yet.")


def render_new_patient_form():
    st.header("➕ Enroll New Patient")
    st.markdown("Enter the patient's discharge information to begin daily monitoring.")

    with st.form("new_patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Full Name", key="name")
            phone = st.text_input("Phone (E.164, e.g., +919876543210)", key="phone")
        with col2:
            age = st.number_input("Age", min_value=18, max_value=120, key="age")
            prior_admissions_30d = st.number_input("Prior Admissions (last 30 days)", min_value=0, max_value=10, key="admissions")
        with col3:
            comorbidity_score = st.number_input("Comorbidity Score (1-10)", min_value=1, max_value=10, key="comorbidity")

        st.markdown("---")
        
        discharge_diagnosis = st.text_input("Primary Discharge Diagnosis (e.g., CHF, COPD, Post-CABG)", key="diagnosis")
        medications = st.text_area("Key Medications (List comma-separated, e.g., Furosemide, Lisinopril, Salbutamol)", key="medications", height=100)
        
        submitted = st.form_submit_button("Enroll Patient", type="primary")

        if submitted:
            if not all([name, phone, discharge_diagnosis, medications]):
                st.error("Please fill in all required fields.")
            else:
                try:
                    data = {
                        "name": name, "phone": phone, "age": age,
                        "prior_admissions_30d": prior_admissions_30d,
                        "comorbidity_score": comorbidity_score,
                        "discharge_diagnosis": discharge_diagnosis,
                        "medications": medications
                    }
                    
                    response = requests.post(f"{FASTAPI_BASE_URL}/api/patients/add", data=data)
                    response.raise_for_status()
                    
                    result = response.json()
                    st.success(f"Patient {result['name']} (ID: {result['patient_id']}) enrolled successfully!")
                    
                    st.cache_data.clear()
                    st.rerun()
                    
                except requests.exceptions.RequestException as e:
                    try:
                        error_detail = response.json().get("detail", str(e))
                    except:
                        error_detail = str(e)
                    st.error(f"Enrollment failed. Error: {error_detail}")

# --- Main Application Layout ---

st.set_page_config(layout="wide", page_title="IVR Clinical Dashboard")

# Navigation Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", ["Alert Monitor", "Enroll New Patient"])
    
    st.markdown("---")
    
    # Manual Call Trigger
    if st.button("📞 Trigger Daily Call Job (Manual)"):
        try:
            response = requests.post(f"{FASTAPI_BASE_URL}/call_patients_job_manual")
            response.raise_for_status()
            st.success("Manual call job triggered successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to trigger job: {e}")

# Page Rendering
if page == "Alert Monitor":
    patient_summary_data = fetch_patient_summary()
    render_alert_monitor(patient_summary_data)
elif page == "Enroll New Patient":
    render_new_patient_form()

# Always show the backend connection status at the bottom (for debugging)
st.sidebar.markdown("---")
st.sidebar.caption(f"FastAPI Backend Target: {FASTAPI_BASE_URL}")
