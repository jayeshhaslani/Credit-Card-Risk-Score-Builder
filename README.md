# Credit Risk Score Builder

A macro-enhanced credit risk scorecard application with a Streamlit frontend and modular backend services. The project preserves the trained scorecard pipeline while presenting it through a polished simulator, macro dashboard, and regulatory assistant surface.

## What this project includes

- A scorecard-based risk simulator that uses the existing trained model artifacts without retraining.
- A macroeconomic dashboard that pulls FRED indicators when available and falls back to defaults otherwise.
- A modular structure that separates UI, preprocessing, prediction, scoring, and external data access.

## Repository structure

- [app.py](app.py) - top-level Streamlit entry point.
- [apps/app.py](apps/app.py) - page shell for the interactive app.
- [apps/pages](apps/pages) - Streamlit pages for the simulator, dashboard, and assistant.
- [src](src) - reusable prediction, preprocessing, scoring, and FRED integration logic.
- [models](models) - pre-trained model artifacts used by the app.
- [Data](Data) - processed and raw datasets used in the analysis workflow.

## Setup

1. Open PowerShell in the project root.
2. Activate the virtual environment:

```powershell
& .\venv\Scripts\Activate.ps1
```

3. Install the project dependencies:

```powershell
pip install -r requirements.txt
```

## Run locally

Use the helper scripts or launch Streamlit directly:

```powershell
.\run_app.ps1
```

or

```powershell
streamlit run app.py
```

## Deployment notes

- The application expects the trained artifacts in [models](models).
- If you have a FRED API key, set it before running the app:

```powershell
$env:FRED_API_KEY = "YOUR_KEY"
```

- The app is designed to work in Streamlit Community Cloud with the repository root as the deployment target.
