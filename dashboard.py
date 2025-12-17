import streamlit as st
import requests
import pandas as pd
import re
from typing import Optional, Dict, Any

# =====================================================
# CONFIG
# =====================================================

FASTAPI_BASE_URL = "https://ivr-clinical-backend.onrender.com"  # UPDATE IF NEEDED

# =====================================================
# HELPERS
# =====================================================

def fetch_data(endpoint: str):
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}{endpoint}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def post_form(endpoint: str, data: Dict[str, Any]):
    try:
        r = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", data=data)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        try:
            st.error(e.response.json().get("detail", "Backend error"))
        except:
            st.error("500 error – check Render backend logs")
        return None
    except Exception as e:
        st.error(e)
        return None


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    return phone

# =====================================================
# ENROLLMENT
# =====================================================

def enrollment_tab():
    st.header("➕ Patient Enrollment")

    with st.form("enroll"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Patient Name")
        phone = col2.text_input("Phone (10 digits)")

        col3, col4 = st.columns(2)
        age = col3.number_input("Age", 18, 120, 60)
        prior = col4.number_input("Prior Admissions (30d)", 0, 5, 0)

        comorbidity = st.slider("Comorbidity Score", 1, 10, 3)

        diagnosis = st.text_area("Discharge Diagnosis")
        meds = st.text_area("Medications (comma separated)")

        submit = st.form_submit_button("Enroll Patient")

        if submit:
            if not all([name, phone, diagnosis, meds]):
                st.error("All fields required")
                return

            phone = normalize_phone(phone)
            if not re.match(r"^\+91\d{10}$", phone):
                st.error("Invalid Indian phone number")
                return

            payload = {
                "name": name,
                "phone": phone,
                "age": age,
                "prior_admissions_30d": prior,
                "comorbidity_score": comorbidity,
                "discharge_diagnosis": diagnosis,
                "medications": meds
            }

            with st.spinner("Enrolling..."):
                res = post_form("/api/patients/add", payload)

            if res and res.get("status") == "success":
                st.success(f"Patient enrolled (ID: {res['patient_id']})")

# =====================================================
# DASHBOARD
# =====================================================

def dashboard_tab():
    st.header("📊 Remote Patient Monitoring")

    if st.button("🔄 Trigger Manual Calls"):
        r = requests.post(f"{FASTAPI_BASE_URL}/call_patients_job_manual")
        if r.ok:
            st.success(f"Calls triggered: {r.json()['count']}")
        else:
            st.error("Call trigger failed")

    data = fetch_data("/api/patients/all_summary")
    if not data:
        st.info("No patients enrolled")
        return

    df = pd.DataFrame(data)
    df["Risk %"] = (df["risk_probability"] * 100).round(1)
    df["Last Report"] = df["last_report"].fillna("").str[:60]

    st.subheader("Patient Risk Summary")

    def color(row):
        if row["risk_level"] == "High":
            return ["background-color:#ffcccc"] * len(row)
        if row["risk_level"] == "Medium":
            return ["background-color:#fff0b3"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df[["id", "name", "risk_level", "Risk %", "doctor_override", "Last Report"]]
        .style.apply(color, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Patient Details")

    pid = st.selectbox("Select Patient", df["id"])

    if pid:
        detail = fetch_data(f"/api/patients/{pid}/history")
        patient = df[df["id"] == pid].iloc[0]

        st.markdown(f"### {patient['name']}")
        st.text(f"Phone: {detail['patient_info']['phone']}")
        st.text(f"Age: {detail['patient_info']['age']}")

        st.markdown("**Diagnosis:**")
        st.write(detail["discharge_summary"]["diagnosis"])

        st.markdown("**Medications:**")
        st.write(detail["discharge_summary"]["medications"])

        st.metric(
            "Risk Level",
            patient["risk_level"],
            f"{patient['Risk %']} % probability"
        )

        st.markdown("#### SHAP Explanation")
        st.dataframe(pd.DataFrame(patient["shap_explanation"]), hide_index=True)

        st.markdown("#### Doctor Intervention")

        notes = st.text_area(
            "Intervention Notes",
            value=patient.get("intervention_notes", "")
        )
        override = st.checkbox("Doctor Override", value=patient["doctor_override"])

        if st.button("Save Intervention"):
            requests.post(
                f"{FASTAPI_BASE_URL}/api/patients/intervene/{pid}",
                json={"notes": notes}
            )
            requests.patch(
                f"{FASTAPI_BASE_URL}/api/patients/update/{pid}",
                json={"doctor_override": override}
            )
            st.success("Updated – refresh dashboard")

        st.markdown("#### Daily Monitoring Log")
        logs = detail["daily_data_log"]
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No daily reports yet")

# =====================================================
# MAIN
# =====================================================

st.set_page_config("IVR Clinical Dashboard", layout="wide")

st.sidebar.title("IVR Monitoring System")
st.sidebar.caption(FASTAPI_BASE_URL)

tab1, tab2 = st.tabs(["📊 Dashboard", "➕ Enrollment"])

with tab1:
    dashboard_tab()

with tab2:
    enrollment_tab()
