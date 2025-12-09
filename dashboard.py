%%writefile dashboard.py
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time

# --- Configuration: CHANGE THIS URL FOR DEPLOYMENT ---
# For local running: 
FASTAPI_BASE_URL = "https://ivr-clinical-backend.onrender.com" 
# For deployment to Streamlit Cloud, CHANGE this to your permanent Render URL:
# FASTAPI_BASE_URL = "https://your-permanent-render-url.onrender.com" 

# --- Helper Functions ---

@st.cache_data(ttl=5) # Cache data for 5 seconds to reduce API calls
def fetch_patient_summary():
    """Fetches the main patient summary data from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/api/patients/all_summary")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to the FastAPI backend at {FASTAPI_BASE_URL}. Please ensure app_backend.py is running. Error: {e}")
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

# --- UI Components ---

def render_alert_monitor(patient_data):
    """Renders the main dashboard table with alerts."""
    st.header("🚨 Patient Tracking & Alert Monitor")

    if not patient_data:
        st.warning("Waiting for data from the backend...")
        return

    df = pd.DataFrame(patient_data)
    
    # Format and sort the data for display
    df['Risk (%)'] = df['risk_probability'].apply(lambda x: f"{x:.1f}%")
    df['Last Call'] = df['last_call'].fillna('Never').apply(lambda x: datetime.fromisoformat(x).strftime('%Y-%m-%d %H:%M') if x != 'Never' else x)
    
    # Sort by risk level
    risk_order = {"High": 3, "Medium": 2, "Low": 1}
    df['risk_rank'] = df['risk_level'].apply(lambda x: risk_order.get(x, 0))
    df = df.sort_values(by=['risk_rank', 'risk_probability'], ascending=[False, False]).drop(columns=['risk_rank'])

    # Apply styling for high risk and intervention
    def style_rows(row):
        style = []
        if row['risk_level'] == 'High':
            style.append('background-color: rgba(255, 0, 0, 0.2);') # Light Red
        if row['doctor_override']:
            style.append('border-left: 5px solid yellow;') # Yellow border for intervention
        return style
    
    # Display the styled table
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
            "doctor_override": "Intervention Logged"
        }
    )

    # --- Detail Section ---
    st.markdown("### 🔍 Patient Details and Intervention")
    selected_id = st.selectbox("Select Patient ID for Details:", df['id'].tolist() if not df.empty else [])

    if selected_id:
        patient_summary = df[df['id'] == selected_id].iloc[0]
        history = fetch_patient_history(selected_id)
        
        st.markdown(f"#### Details for {patient_summary['name']} ({selected_id})")

        # Layout for Risk/Explanation/Intervention
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Current Risk Level", value=patient_summary['risk_level'])
            st.metric(label="Readmission Probability", value=patient_summary['Risk (%)'])
            st.markdown(f"**Last Report:** {patient_summary['last_report']}")

        with col2:
            st.markdown("**ML Risk Drivers (SHAP Explanation)**")
            if patient_summary['shap_explanation']:
                shap_df = pd.DataFrame(patient_summary['shap_explanation'])
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
            else:
                st.info("No ML explanation available yet.")
            
            # Intervention Logging
            st.markdown("---")
            st.markdown("#### Doctor Intervention Log")
            
            if patient_summary['doctor_override']:
                st.success(f"Intervention previously logged. Notes: {patient_summary['intervention_notes']}")
            
            intervention_notes = st.text_area("Log Intervention/Follow-up Notes:", height=100)
            
            if st.button(f"Log Intervention for {selected_id}", type="primary"):
                if intervention_notes:
                    log_intervention(selected_id, intervention_notes)
                    st.toast("Intervention logged successfully!", icon='✅')
                    time.sleep(1) # Wait for cache to expire and refresh
                    st.rerun()
                else:
                    st.warning("Please enter intervention notes.")

        # Historical Data Section
        st.markdown("---")
        st.markdown("### 📈 Historical Daily Reports")
        if history and history.get('daily_data_log'):
            history_df = pd.DataFrame(history['daily_data_log'])
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No historical reports yet.")


def render_new_patient_form():
    """Renders the form to enroll a new patient."""
    st.header("➕ Enroll New Patient")
    st.markdown("Enter the patient's discharge information to begin daily monitoring.")

    with st.form("new_patient_form"):
        # Personal/Clinical Data
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
        
        # Discharge Summary
        discharge_diagnosis = st.text_input("Primary Discharge Diagnosis (e.g., CHF, COPD, Post-CABG)", key="diagnosis")
        medications = st.text_area("Key Medications (List comma-separated, e.g., Furosemide, Lisinopril, Salbutamol)", key="medications", height=100)
        
        submitted = st.form_submit_button("Enroll Patient", type="primary")

        if submitted:
            # Basic client-side validation
            if not all([name, phone, discharge_diagnosis, medications]):
                st.error("Please fill in all required fields.")
            else:
                try:
                    data = {
                        "name": name,
                        "phone": phone,
                        "age": age,
                        "prior_admissions_30d": prior_admissions_30d,
                        "comorbidity_score": comorbidity_score,
                        "discharge_diagnosis": discharge_diagnosis,
                        "medications": medications
                    }
                    
                    response = requests.post(f"{FASTAPI_BASE_URL}/api/patients/add", data=data)
                    response.raise_for_status()
                    
                    result = response.json()
                    st.success(f"Patient {result['name']} (ID: {result['patient_id']}) enrolled successfully!")
                    
                    # Clear form and refresh cache
                    st.cache_data.clear()
                    time.sleep(1) 
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
st.sidebar.caption(f"FastAPI Backend: {FASTAPI_BASE_URL}")