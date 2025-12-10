
# 1. Backend: Runs the FastAPI app using Uvicorn on port 8000
web: uvicorn app_backend:app --host 0.0.0.0 --port 8000

# 2. Frontend: Runs the Streamlit dashboard on port 8501
frontend: streamlit run dashboard.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
