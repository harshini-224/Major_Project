# -*- coding: utf-8 -*-
"""dashboard.py: Streamlit Patient Monitoring Dashboard"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os
import time

# --- Configuration ---
# Set the initial backend URL. This will be updated by /get_public_url
BACKEND_BASE_URL = os.environ.get("BACKEND_URL_OVERRIDE", "http://localhost:8000")

# --- Helper Functions ---
@st.cache_data(ttl=5) # Cache data for 5 seconds
def fetch_patient_summary(url):
    try:
        response = requests.get(f"{url}/api/patients/all_summary", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patient summary from backend: {e}")
        st.stop()
    return []

def trigger_manual_call(url):
    try:
        response = requests.post(f"{url}/call_patients_job_manual", timeout=5)
        response.raise_for_status()
        st.success("Manual call job triggered! Check backend logs for call status.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error triggering call job: {e}")

def log_intervention(url, patient_id, notes):
    try:
        response = requests.post(
            f"{url}/api/patients/intervene/{patient_id}", 
            json={"notes": notes}, 
            timeout=5
        )
        response.raise_for_status()
        st.success(f"Intervention logged for {patient_id}. Patient status is now overridden.")
        # Clear the cache to force a refresh
        st.cache_data.clear()
        time.sleep(1) # Pause for success message display
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Error logging intervention: {e}")

def get_risk_color(level):
    if level == "High": return "red"
    if level == "Medium": return "orange"
    return "green"

# --- Layout Functions ---
def render_main_dashboard(data, backend_url):
    st.title("🏥 Patient Telemonitoring Dashboard")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.write("Current Risk Overview")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    st.subheader("High and Medium Risk Patients")
    
    # Filter and sort data for the main view
    df = pd.DataFrame(data)
    
    if df.empty:
        st.info("No patients currently enrolled in the system.")
        return

    # Prepare DataFrame for Display
    df_display = df.copy()
    df_display['Risk %'] = df_display['risk_probability'].apply(lambda x: f"{x:.1f}%")
    df_display['Last Report'] = df_display['last_report'].apply(lambda x: x[:70] + '...' if len(x) > 70 else x)
    df_display['Last Call'] = df_display['last_call'].fillna("Never")
    
    # Display High/Medium Risk Patients first
    high_medium_risk = df_display[df_display['risk_level'].isin(['High', 'Medium'])]
    low_risk = df_display[df_display['risk_level'] == 'Low']
    
    
    # --- High/Medium Table ---
    if not high_medium_risk.empty:
        st.markdown("**Urgent Attention Required**")
        st.dataframe(
            high_medium_risk[['id', 'name', 'risk_level', 'Risk %', 'Last Call', 'Last Report', 'doctor_override']],
            hide_index=True,
            column_config={
                "doctor_override": st.column_config.CheckboxColumn("Doctor Override", disabled=True),
            },
            use_container_width=True
        )
    else:
        st.success("No High or Medium risk patients currently detected. All clear!")

    # --- Low Risk Table ---
    if not low_risk.empty:
        st.markdown("---")
        st.markdown("**Low Risk Patients (Scheduled Monitoring)**")
        st.dataframe(
            low_risk[['id', 'name', 'risk_level', 'Risk %', 'Last Call']],
            hide_index=True,
            use_container_width=True
        )
        
    st.markdown("---")
    st.subheader("📞 Monitoring Job Control")
    st.write(f"Daily automated calls are scheduled for **{os.environ.get('DAILY_CALL_HOUR_IST', '10')}:00 IST**.")
    if st.button("📞 Trigger Daily Call Job (Manual Test)"):
        trigger_manual_call(backend_url)

def render_patient_details(patient_id, backend_url):
    st.title(f"Patient Details for {patient_id}")
    
    try:
        response = requests.get(f"{backend_url}/api/patients/{patient_id}/history", timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patient history: {e}")
        return

    patient_info = data['patient_info']
    daily_data_log = data['daily_data_log']
    discharge_summary = data['discharge_summary']
    
    # Fetch current risk data from summary (since history doesn't contain it)
    current_summary = next((p for p in fetch_patient_summary(backend_url) if p['id'] == patient_id), {})
    if not current_summary:
        st.warning("Could not find real-time risk data for this patient.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.header(f"Details for {patient_info['name']} ({patient_id})")
        st.markdown(f"**Current Risk Level:** <span style='color:{get_risk_color(current_summary['risk_level'])}; font-size: 24px;'>{current_summary['risk_level']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Readmission Probability:** **{current_summary['risk_probability']:.1f}%**")
        st.markdown(f"**Last Report:** {current_summary['last_report']}")
        st.markdown(f"**Age:** {patient_info['age']} | **Phone:** {patient_info['phone']}")
        
        st.subheader("Discharge Summary")
        st.info(f"**Diagnosis:** {discharge_summary['diagnosis']}\n\n**Medications:** {discharge_summary['medications']}")

    with col2:
        st.subheader("ML Risk Drivers (SHAP Explanation)")
        if current_summary['shap_explanation']:
            explanation_df = pd.DataFrame(current_summary['shap_explanation'])
            explanation_df.columns = ['Feature', 'Impact Score']
            st.dataframe(explanation_df, hide_index=True)
        else:
            st.info("No ML explanation available yet.")

        st.subheader("Doctor Intervention Log")
        intervention_notes = st.text_area(
            "Log Intervention/Follow-up Notes:", 
            value=current_summary['intervention_notes'] if current_summary['doctor_override'] else ""
        )
        if st.button(f"Log Intervention for {patient_id}"):
            log_intervention(backend_url, patient_id, intervention_notes)

    st.markdown("---")
    st.subheader("📅 Historical Daily Reports")
    
    if daily_data_log:
        # Create a display DataFrame from the log
        log_df = pd.DataFrame(daily_data_log)
        log_df['Risk %'] = (log_df['ml_prediction'] * 100).apply(lambda x: f"{x:.1f}%")
        log_df['Adherence'] = log_df['adherence'].apply(lambda x: 'Yes' if x == 1 else 'No')
        
        # Select and rename columns for display
        st.dataframe(
            log_df[['date', 'risk_level', 'Risk %', 'Adherence', 'symptom_report']],
            hide_index=True,
            column_order=('date', 'risk_level', 'Risk %', 'Adherence', 'symptom_report'),
            column_config={
                'date': 'Report Date',
                'risk_level': 'Risk Level',
                'symptom_report': 'Symptom Report'
            },
            use_container_width=True
        )
    else:
        st.info("No historical reports yet.")
    
    st.markdown("---")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.view = 'dashboard'
        st.rerun()

def render_new_patient_form(backend_url):
    st.title("👤 Enroll New Patient")
    st.write("Enter patient details to begin monitoring and scheduling daily IVR calls.")

    with st.form("new_patient_form"):
        name = st.text_input("Full Name", key="name")
        col1, col2 = st.columns(2)
        with col1:
            phone = st.text_input("Phone (E.164, e.g., +919876543210)", key="phone")
            age = st.number_input("Age", min_value=18, max_value=120, key="age")
        with col2:
            prior_admissions_30d = st.number_input("Prior Admissions (last 30 days)", min_value=0, key="prior_admissions_30d")
            # Corrected max_value to 8 for consistency with ML model
            comorbidity_score = st.number_input("**Comorbidity Score (1-8)**", min_value=1, max_value=8, key="comorbidity")
            
        discharge_diagnosis = st.text_area("Primary Discharge Diagnosis (e.g., CHF, COPD, Post-CABG)", key="diagnosis")
        medications = st.text_area("Key Medications (List comma-separated, e.g., Furosemide, Lisinopril, Salbutamol)", key="medications")
        
        submitted = st.form_submit_button("Enroll Patient")
        
        if submitted:
            if not all([name, phone, age, discharge_diagnosis, medications]):
                st.error("Please fill in all required fields.")
            else:
                try:
                    form_data = {
                        "name": name, "phone": phone, "age": age, 
                        "prior_admissions_30d": prior_admissions_30d, 
                        "comorbidity_score": comorbidity_score, 
                        "discharge_diagnosis": discharge_diagnosis, 
                        "medications": medications
                    }
                    
                    response = requests.post(
                        f"{backend_url}/api/patients/add", 
                        data=form_data, 
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    st.success(f"Patient {name} ({response.json()['patient_id']}) enrolled successfully!")
                    st.cache_data.clear() # Clear cache to refresh dashboard view
                except requests.exceptions.RequestException as e:
                    try:
                        error_detail = response.json().get('detail', str(e))
                        st.error(f"Enrollment failed: {error_detail}")
                    except:
                        st.error(f"Enrollment failed: {e}")

# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Telemonitoring Dashboard")

    # 1. Get the Public URL from the Backend
    global BACKEND_BASE_URL
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/get_public_url", timeout=5)
        response.raise_for_status()
        BACKEND_BASE_URL = response.json().get('public_url')
        st.session_state.backend_url = BACKEND_BASE_URL
    except requests.exceptions.RequestException:
        st.warning("Could not fetch public URL. Using default: http://localhost:8000. IVR calls may fail.")
        st.session_state.backend_url = BACKEND_BASE_URL
        
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    view_options = ["Dashboard", "Enroll New Patient"]
    selected_view = st.sidebar.radio("Go to:", view_options)
    
    # Check for patient detail click
    if 'view' not in st.session_state:
        st.session_state.view = 'dashboard'
    if 'selected_patient_id' not in st.session_state:
        st.session_state.selected_patient_id = None
        
    # Handle view state
    if selected_view == "Enroll New Patient":
        st.session_state.view = 'enroll'
        st.session_state.selected_patient_id = None
    elif st.session_state.view == 'dashboard' and selected_view == 'Dashboard':
        pass # Stay in dashboard view
    
    # --- Render Views ---
    if st.session_state.view == 'enroll':
        render_new_patient_form(st.session_state.backend_url)
        
    elif st.session_state.view == 'details':
        render_patient_details(st.session_state.selected_patient_id, st.session_state.backend_url)
        
    else: # Default to dashboard
        all_patients_data = fetch_patient_summary(st.session_state.backend_url)
        render_main_dashboard(all_patients_data, st.session_state.backend_url)
        
        # Add buttons to view patient details at the bottom of the dashboard
        if all_patients_data:
            st.markdown("---")
            st.subheader("View Detailed History")
            cols = st.columns(4)
            for i, patient in enumerate(all_patients_data):
                if cols[i % 4].button(f"Details for {patient['name']} ({patient['id']})"):
                    st.session_state.selected_patient_id = patient['id']
                    st.session_state.view = 'details'
                    st.rerun()

if __name__ == "__main__":
    main()
