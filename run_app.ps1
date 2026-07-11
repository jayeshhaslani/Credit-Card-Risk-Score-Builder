# Launch the Streamlit application from the repository root.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
& .\venv\Scripts\Activate.ps1
streamlit run app.py
