# -*- coding: utf-8 -*-
"""dashboard.py: Streamlit Frontend Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os

# --- 0) Configuration ---
# Use environment variable if available (e.g., in Render deployment)
BACKEND_BASE_URL = os.environ.get("BACKEND_URL_OVERRIDE", "https://ivr-clinical-backend.onrender.com")

# --- 1) API Functions ---
def fetch_patient_summary():
    """Fetches the summary data for all patients."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/patients/all_summary")
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patient summary from backend: {e}")
        return None

def fetch_patient_history(patient_id):
    """Fetches detailed history and discharge summary for a single patient."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/patients/{patient_id}/history")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patient history: {e}")
        return None

# --- 2) Navigation and State Management ---
# Initialize session state for tracking the current page and patient selection
if 'page' not in st.session_state:
    st.session_state['page'] = 'Dashboard' # Default starting page

if 'selected_patient_id' not in st.session_state:
    st.session_state['selected_patient_id'] = None 

# CRITICAL FIX: Replaced st.experimental_rerun() with the correct st.rerun()
def set_page_state(page_name, patient_id=None):
    """Sets the page and selected patient ID, then triggers a rerun."""
    st.session_state['page'] = page_name
    st.session_state['selected_patient_id'] = patient_id
    # Clear confirmation state when changing pages
    if 'delete_confirm' in st.session_state:
        del st.session_state['delete_confirm'] 
    
    st.rerun() # This is the corrected function call

# --- 3) Rendering Functions ---

# --- A. Enrollment Page ---
def render_enrollment_page():
    st.title("➕ Enroll New Patient")
    st.markdown("Enter the patient's data to start automated monitoring.")
    
    with st.form("new_patient_form"):
        st.subheader("Patient Demographics")
        name = st.text_input("Full Name", required=True)
        phone = st.text_input("Phone Number (E.164 format, e.g., +919876543210)", required=True)
        
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", min_value=18, max_value=100, value=65)
        prior_admissions_30d = col2.number_input("Prior Admissions (last 30 days)", min_value=0, max_value=5, value=0)
        comorbidity_score = col3.slider("Comorbidity Score (1-10)", min_value=1, max_value=10, value=4)
        
        st.subheader("Discharge Details")
        discharge_diagnosis = st.text_input("Primary Discharge Diagnosis (e.g., CHF, COPD)")
        medications = st.text_area("Key Medications (e.g., Furosemide, Lisinopril)")
        
        submitted = st.form_submit_button("Enroll Patient")
        
        if submitted:
            # Prepare data for POST request
            data = {
                "name": name, "phone": phone, "age": age, 
                "prior_admissions_30d": prior_admissions_30d, "comorbidity_score": comorbidity_score,
                "discharge_diagnosis": discharge_diagnosis, "medications": medications
            }
            
            try:
                # Send enrollment data to the backend
                response = requests.post(f"{BACKEND_BASE_URL}/api/patients/add", data=data)
                response.raise_for_status()
                
                result = response.json()
                st.success(f"✅ Patient {result['name']} enrolled successfully! ID: {result['patient_id']}")
                st.balloons()
                
            except requests.exceptions.RequestException as e:
                try:
                    error_detail = response.json().get('detail', str(e))
                except:
                    error_detail = str(e)
                st.error(f"Enrollment failed: {error_detail}")
            except NameError:
                 st.error("Backend URL is not defined. Check configuration.")


# --- B. Patient Details Page ---
def render_patient_details(patient_id):
    patient_data = fetch_patient_history(patient_id)
    if not patient_data:
        st.error(f"Could not load details for patient {patient_id}.")
        # Use on_click callback for safer navigation
        if st.button("⬅️ Back to List", on_click=set_page_state, args=('Dashboard',)):
            pass 
        return

    st.session_state['patient_data'] = patient_data # Store for easy access
    patient_info = patient_data['patient_info']
    
    st.title(f"👤 Patient Details: {patient_info['name']} ({patient_id})")

    # --- ACTION BUTTONS (Back, Delete, Call) ---
    col1, col2, col3, col4 = st.columns([1, 1.5, 1, 5]) 

    with col1:
        # Use on_click callback for safer navigation
        st.button("⬅️ Back to List", on_click=set_page_state, args=('Dashboard',))
            
    with col2:
        if st.button("🗑️ Delete Patient", type="primary"):
            st.session_state['delete_confirm'] = True
            
    with col3:
        if st.button("📞 Call Patient", type="secondary"):
            try:
                response = requests.post(f"{BACKEND_BASE_URL}/call_patient_manual/{patient_id}")
                response.raise_for_status()
                st.success(f"📞 Call initiated for {patient_info['name']}!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error initiating call: {e}")

    # --- Deletion Confirmation Logic ---
    if st.session_state.get('delete_confirm'):
        st.warning(f"⚠️ Are you sure you want to permanently delete patient {patient_info['name']}? This action cannot be undone.")
        
        colA, colB = st.columns([1, 7])
        with colA:
            if st.button("Yes, Delete Permanently", type="error"):
                try:
                    response = requests.delete(f"{BACKEND_BASE_URL}/api/patients/delete/{patient_id}")
                    response.raise_for_status()
                    
                    st.success(f"Patient {patient_id} deleted successfully.")
                    # Navigate back to the list using the callback approach
                    set_page_state('Dashboard') 
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error deleting patient: {e}")
                    st.session_state['delete_confirm'] = False
        st.markdown("---") 

    st.subheader("Historical Daily Reports")
    daily_data = pd.DataFrame(patient_data['daily_data_log'])
    
    if not daily_data.empty:
        # Format the DataFrame for display
        daily_data['date'] = pd.to_datetime(daily_data['date']).dt.strftime('%Y-%m-%d %H:%M')
        daily_data = daily_data.set_index('date')
        
        # Display the table and the latest report card
        latest_report = daily_data.iloc[0]

        st.markdown(f"**Latest Risk Level:** :red[**{latest_report['risk_level']}**] ({latest_report['ml_prediction']*100:.1f}%)")
        st.markdown(f"**Latest Report:** *{latest_report['symptom_report']}*")

        with st.expander("View Full Log"):
            st.dataframe(daily_data, height=300)
    else:
        st.info("No daily reports recorded yet.")
        
    st.subheader("Discharge Summary")
    summary = patient_data['discharge_summary']
    st.json(summary)


# --- C. Main Dashboard (Patient List) ---
def render_main_dashboard():
    st.title("📋 Patient List & Risk Summary")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun() # Changed to st.rerun()
        
    patient_summary = fetch_patient_summary()
    
    if patient_summary is None:
        st.error("Cannot connect to the backend API. Please ensure the FastAPI server is running.")
        return
        
    if not patient_summary:
        st.info("No patients enrolled yet. Enroll a new patient to begin monitoring.")
        return

    df = pd.DataFrame(patient_summary)
    
    # Format and display the DataFrame
    df['Risk'] = df.apply(lambda row: f"{row['risk_level']} ({row['risk_probability']:.1f}%)", axis=1)
    
    # Prepare the list for display
    # Create an interactive table
    st.markdown("---")
    
    # Display headers
    col_id, col_name, col_risk, col_last_call, col_button = st.columns([1, 2, 1.5, 2, 1])
    col_id.markdown("**ID**")
    col_name.markdown("**Name**")
    col_risk.markdown("**Risk**")
    col_last_call.markdown("**Last Contact**")
    col_button.markdown("**Actions**")

    # Display each patient with a button to view details
    for index, row in df.iterrows():
        col_id, col_name, col_risk, col_last_call, col_button = st.columns([1, 2, 1.5, 2, 1])
        
        risk_color = "red" if row['risk_level'] == 'High' else "orange" if row['risk_level'] == 'Medium' else "green"
        
        col_id.markdown(f"**{row['id']}**")
        col_name.markdown(f"**{row['name']}**")
        col_risk.markdown(f":{risk_color}[{row['Risk']}]")
        col_last_call.markdown(f"*{row['last_call'] or 'N/A'}*")

        with col_button:
            # Use on_click callback for stable navigation
            st.button(
                "View Details", 
                key=f"view_{row['id']}",
                on_click=set_page_state,
                args=('Patient Details', row['id'])
            )
            
    st.markdown("---")
    
    # --- Monitoring Job Control (Manual Call Trigger) ---
    st.subheader("Monitoring Job Control (Manual Test)")
    
    if st.button("📞 Trigger Daily Call Job (Bulk Test)"):
        try:
            requests.post(f"{BACKEND_BASE_URL}/call_patients_job_manual")
            st.success("Bulk call job initiated! Check your phone(s).")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to trigger job: {e}")


# --- 4) MAIN EXECUTION FLOW ---

# --- Sidebar Navigation ---
st.sidebar.title("🏥 RPM Dashboard")
st.sidebar.markdown("---")

# The radio button sets the 'page' state variable directly
page_selection = st.sidebar.radio(
    "Navigation",
    ("Dashboard", "Enroll New Patient"),
    # Set the index based on the current page in session state
    index=0 if st.session_state['page'] in ('Dashboard', 'Patient Details') else 1
)

# CRITICAL: If the radio button changes the selection, switch the state immediately
if page_selection != st.session_state['page']:
    set_page_state(page_selection) 
    
st.sidebar.markdown("---")

# --- Main Page Rendering Logic ---

if st.session_state['page'] == 'Dashboard':
    render_main_dashboard()
    
elif st.session_state['page'] == 'Enroll New Patient':
    render_enrollment_page()

# 'Patient Details' is not a sidebar option, but a view triggered by the Dashboard.
elif st.session_state['selected_patient_id']:
    render_patient_details(st.session_state['selected_patient_id'])

