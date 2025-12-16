import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import re
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
# CRITICAL: Replace this placeholder with your actual Render FastAPI service URL
# Example: "https://ivr-clinical-backend.onrender.com"
FASTAPI_BASE_URL: str = "https://ivr-clinical-backend.onrender.com" # <<<--- UPDATE THIS URL

# --- HELPER FUNCTIONS ---

def fetch_data(endpoint: str) -> Optional[Dict[str, Any]]:
    """Fetches data from the FastAPI backend."""
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend at {url}. Ensure the Render service is running and the URL is correct.")
        return None

def post_data(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Posts form-data to the FastAPI backend and handles non-JSON errors gracefully."""
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    try:
        # FastAPI's endpoint uses Form(...), so we use data=data for form-data encoding
        response = requests.post(url, data=data) 
        response.raise_for_status() # Raises HTTPError for 4xx or 5xx status codes
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        # --- CORRECTED ERROR HANDLING ---
        status_code = e.response.status_code
        detail_message = f"HTTP Status Code: {status_code}. "
        
        try:
            # Attempt to decode JSON detail (works for 400 Bad Request/Validation errors)
            detail_message += e.response.json().get('detail', 'Unknown error from API.')
        except requests.exceptions.JSONDecodeError:
            # Fallback for 500 Internal Server Errors where the backend returns non-JSON/HTML
            detail_message += "Backend returned a non-JSON error response. **CRITICAL: Check your Render FastAPI Backend Logs for the full 500 Server Error traceback!**"
            
        st.error(f"Enrollment Failed: {detail_message}")
        return None
        # --- END CORRECTED ERROR HANDLING ---
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network or connection error to backend: {e}")
        return None

def normalize_phone_number(phone: str) -> str:
    """Normalizes a phone number string to E.164 format (+91XXXXXXXXXX)."""
    clean_phone = re.sub(r'[^\d+]', '', phone)
    if clean_phone.startswith('+91') and len(clean_phone) == 13: 
        return clean_phone
    if len(clean_phone) == 10: 
        return f"+91{clean_phone}"
    return clean_phone 

# --- STREAMLIT DASHBOARD FUNCTIONS ---

def render_enrollment_tab():
    """Renders the patient enrollment form."""
    st.header("➕ New Patient Enrollment")
    st.warning("Please ensure the phone number is a valid 10-digit Indian number for Twilio calls.")

    with st.form("enrollment_form"):
        # --- PATIENT INFO ---
        st.subheader("Patient Demographics")
        col1, col2 = st.columns(2)
        
        name = col1.text_input("Full Name", key="name_input")
        phone_input = col2.text_input("Phone Number (10 digits, e.g., 9876543210)", key="phone_input")
        
        col3, col4 = st.columns(2)
        age = col3.number_input("Age (Years)", min_value=18, max_value=120, value=60, key="age_input")
        prior_admissions_30d = col4.number_input("Prior Admissions (Last 30 Days)", min_value=0, max_value=5, value=0, key="adm_input")
        
        # --- CLINICAL INFO ---
        st.subheader("Clinical Data (for Risk Model)")
        
        comorbidity_score = st.slider("Comorbidity Score (1=Low Risk, 10=High Risk)", min_value=1, max_value=10, value=3, key="com_input")
        
        discharge_diagnosis = st.text_area("Primary Discharge Diagnosis", placeholder="e.g., Congestive Heart Failure (CHF)", key="diag_input")
        medications = st.text_area("Discharge Medications (Comma separated list)", placeholder="e.g., Furosemide 40mg, Lisinopril 10mg", key="meds_input")

        # --- SUBMIT BUTTON ---
        submitted = st.form_submit_button("Enroll Patient and Start Monitoring")

        if submitted:
            # 1. Validation
            if not all([name, phone_input, discharge_diagnosis, medications]):
                st.error("Please fill in all required fields.")
                return
            
            # Normalize phone number to E.164 format (+91...)
            phone_e164 = normalize_phone_number(phone_input)

            if not re.match(r'^\+91\d{10}$', phone_e164):
                st.error("Invalid phone number format. Please enter a 10-digit number.")
                return

            # 2. Data Preparation
            enrollment_data = {
                "name": name,
                "phone": phone_e164,
                "age": age,
                "prior_admissions_30d": prior_admissions_30d,
                "comorbidity_score": comorbidity_score,
                "discharge_diagnosis": discharge_diagnosis,
                "medications": medications
            }

            # 3. API Call
            with st.spinner("Enrolling patient..."):
                response = post_data("/api/patients/add", data=enrollment_data)

            # 4. Result Handling
            if response and response.get("status") == "success":
                st.success(f"✅ Patient **{name}** enrolled successfully!")
                st.info(f"Assigned ID: **{response['patient_id']}**")
                
                # Clear form fields on success
                st.session_state["name_input"] = ""
                st.session_state["phone_input"] = ""
                st.session_state["age_input"] = 60
                st.session_state["adm_input"] = 0
                st.session_state["com_input"] = 3
                st.session_state["diag_input"] = ""
                st.session_state["meds_input"] = ""
            elif response:
                 # This block handles successful JSON errors (e.g., 400 Bad Request)
                 # The post_data function already displays the error message.
                 pass

def render_dashboard_tab():
    """Renders the patient monitoring dashboard."""
    st.header("📊 Remote Patient Monitoring Dashboard")
    st.caption("Risk predictions are updated after every automated patient call.")

    col_btn, col_spacer = st.columns([1, 4])
    if col_btn.button("🔄 Refresh Data & Trigger Manual Call Job"):
        with st.spinner("Triggering manual call job on backend..."):
            try:
                # Manual trigger uses a POST request
                response = requests.post(f"{FASTAPI_BASE_URL}/call_patients_job_manual")
                response.raise_for_status()
                st.success(f"Manual call job triggered (will only call {response.json().get('count', 0)} patients). Refreshing dashboard...")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to trigger call job: {e}")
                
        st.cache_data.clear() # Clear cache to force refresh

    data = fetch_data("/api/patients/all_summary")

    if data is not None and data:
        df = pd.DataFrame(data)
        
        # Display Summary Table
        st.subheader("Current Patient Risk Summary")
        
        # Format the DataFrame for better display
        df['Risk (%)'] = (df['risk_probability']).round(1).astype(str) + '%'
        df['Last Report'] = df['symptom_report'].apply(lambda x: x[:70] + "..." if len(x) > 70 else x)
        
        # Determine the table color based on risk level
        def highlight_risk(s):
            if s['risk_level'] == 'High':
                return ['background-color: #ffcccc'] * len(s) # Light Red
            elif s['risk_level'] == 'Medium':
                return ['background-color: #ffe0b2'] * len(s) # Light Orange
            else:
                return [''] * len(s)

        display_cols = ['id', 'name', 'risk_level', 'Risk (%)', 'last_call', 'doctor_override', 'Last Report']
        
        st.dataframe(
            df[display_cols].style.apply(highlight_risk, axis=1), 
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        
        # --- DETAILED VIEW ---
        st.subheader("Detailed Patient View & Intervention")
        
        patient_names = {p['id']: f"{p['name']} (ID: {p['id']})" for p in data}
        selected_id = st.selectbox("Select Patient for Detail View", options=list(patient_names.keys()), format_func=lambda x: patient_names[x])
        
        if selected_id:
            detail_data = fetch_data(f"/api/patients/{selected_id}/history")
            selected_patient = next(item for item in data if item["id"] == selected_id)

            if detail_data:
                info = detail_data['patient_info']
                summary = detail_data['discharge_summary']
                
                # --- Patient & Discharge Info ---
                st.markdown(f"### Details for {info['name']}")
                st.text(f"Age: {info['age']} | Phone: {info['phone']}")
                st.markdown(f"**Discharge Diagnosis:** {summary['diagnosis']} (Date: {summary['date']})")
                st.markdown(f"**Medications:** {summary['medications']}")

                # --- Current Risk & SHAP Explanation ---
                st.markdown("#### Current Risk Assessment")
                st.metric(
                    label="Current Risk Level", 
                    value=selected_patient['risk_level'], 
                    delta=f"{selected_patient['Risk (%)']} readmission probability"
                )
                
                st.markdown("**Top Factors Influencing Risk Prediction:**")
                
                shap_df = pd.DataFrame(selected_patient['shap_explanation'])
                st.dataframe(shap_df, hide_index=True, use_container_width=True)

                # --- Intervention Logging ---
                st.markdown("#### Doctor Intervention Log")
                
                current_notes = selected_patient.get('intervention_notes', 'No intervention logged.')
                
                # Show current status and notes
                if selected_patient['doctor_override']:
                    st.success(f"Intervention Active: {current_notes}")
                else:
                    st.info("No active intervention override.")
                    
                with st.form("intervention_form"):
                    notes = st.text_area("Intervention Notes (e.g., 'Called patient, increased Furosemide dose.')", key="intervention_notes_input")
                    override_toggle = st.checkbox("Set 'Doctor Override' to True (Pauses daily automated call)", value=selected_patient['doctor_override'])
                    
                    col_log, col_clear = st.columns(2)
                    
                    if col_log.form_submit_button("Log Intervention & Override"):
                        try:
                            # 1. Log Intervention Notes (POST request with JSON body)
                            update_notes_response = requests.post(
                                f"{FASTAPI_BASE_URL}/api/patients/intervene/{selected_id}",
                                json={"notes": notes}
                            )
                            update_notes_response.raise_for_status()
                            
                            # 2. Update Doctor Override Flag (PATCH request with JSON body)
                            update_override_response = requests.patch(
                                f"{FASTAPI_BASE_URL}/api/patients/update/{selected_id}",
                                json={"doctor_override": override_toggle}
                            )
                            update_override_response.raise_for_status()
                            
                            st.success("Intervention and Override status updated successfully! Click refresh to view changes.")
                            st.cache_data.clear()
                            st.rerun() # Rerun to refresh the UI immediately
                            
                        except requests.exceptions.HTTPError as e:
                            st.error(f"Intervention Failed: HTTP {e.response.status_code}. Detail: {e.response.json().get('detail', 'Check Backend Logs')}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error during intervention log: {e}")


                # --- Daily Log ---
                st.markdown("#### Daily Monitoring Log")
                if detail_data['daily_data_log']:
                    log_df = pd.DataFrame(detail_data['daily_data_log'])
                    log_df['date'] = pd.to_datetime(log_df['date'])
                    log_df = log_df.sort_values(by='date', ascending=False)
                    
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No daily reports available for this patient.")

    elif data is not None:
        st.info("No patients are currently enrolled. Please use the 'Patient Enrollment' tab to add a patient.")


# --- MAIN APP EXECUTION ---

st.set_page_config(
    page_title="IVR Clinical Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("IVR Monitoring System")
st.sidebar.markdown(f"**Backend:** `{FASTAPI_BASE_URL}`")
st.sidebar.caption("Ensure the backend service is running on Render and this URL is correct.")

# Tabs
tab1, tab2 = st.tabs(["🏥 Monitoring Dashboard", "➕ Patient Enrollment"])

with tab1:
    render_dashboard_tab()

with tab2:
    render_enrollment_tab()
