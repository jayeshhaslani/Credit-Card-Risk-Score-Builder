# Credit Card Risk Score Builder

A production-style **Credit Risk Intelligence Platform** that predicts the probability of borrower default using a **Weight of Evidence (WOE) Scorecard**, **Logistic Regression**, and **macroeconomic indicators from the Federal Reserve (FRED API)**. The project includes an interactive **Streamlit** application for real-time credit risk simulation and portfolio analysis.

---

## Project Overview

Financial institutions must accurately assess the creditworthiness of loan applicants while accounting for both borrower characteristics and prevailing economic conditions.

This project builds an end-to-end **credit risk scorecard** inspired by traditional banking risk models by combining:

- Customer credit profile
- Loan characteristics
- Macroeconomic indicators
- Statistical scorecard modeling
- Interactive risk simulation

Unlike many machine learning credit scoring projects, this implementation follows the **traditional banking scorecard approach**, emphasizing interpretability and regulatory-friendly modeling.

---

## Features

- WOE (Weight of Evidence) Feature Engineering
- Information Value (IV) Analysis
- Logistic Regression Scorecard
- Credit Score Generation (300–850)
- Probability of Default Prediction
- Risk Band Classification
- Decision Engine
  - Approve
  - Manual Review
  - Reject
- Federal Reserve Economic Data (FRED) Integration
- Interactive Streamlit Dashboard
- Modular Project Architecture

---

## Project Architecture

```
Raw Lending Club Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
WOE Binning
        │
        ▼
Information Value Analysis
        │
        ▼
Logistic Regression Scorecard
        │
        ▼
Probability of Default
        │
        ▼
Credit Score Generation
        │
        ▼
Decision Engine
        │
        ▼
Streamlit Application
```

---

## Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-Learn
- Logistic Regression

### Feature Engineering

- ScorecardPy
- Weight of Evidence (WOE)
- Information Value (IV)

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Plotly
- Streamlit

### External Data

- Federal Reserve Economic Data (FRED API)

---

## Dataset

Primary dataset:

**Lending Club Loan Dataset (2007–2018)**

The model uses historical Lending Club loan records including:

- Loan Amount
- Interest Rate
- Employment Length
- Debt-to-Income Ratio
- Home Ownership
- Credit History
- FICO Score
- Revolving Credit Usage
- Loan Purpose
- Loan Status

Macroeconomic variables are enriched using:

- Federal Funds Rate
- Consumer Price Index (CPI)
- Unemployment Rate

---

## Machine Learning Pipeline

### 1. Data Cleaning

- Missing value treatment
- Variable selection
- Target engineering

---

### 2. Feature Engineering

Weight of Evidence transformation

Information Value calculation

Optimal binning

---

### 3. Model

Logistic Regression

Reason for selection:

- Highly interpretable
- Widely used in banking
- Regulatory friendly
- Stable scorecard implementation

---

### 4. Macroeconomic Enhancement

The baseline scorecard is enriched with live economic indicators from the Federal Reserve.

Additional variables include:

- Federal Funds Rate
- CPI
- Unemployment Rate

This allows the model to account for changing economic environments.

---

## Model Performance

| Model | ROC-AUC | KS Statistic |
|---------|---------|-------------|
| Baseline Scorecard | **0.700** | **0.300** |
| Macro Enhanced Scorecard | **0.712** | **0.317** |

Adding macroeconomic indicators improved predictive performance over the baseline scorecard.

---

## Streamlit Application

The project includes an interactive Streamlit application with multiple pages.

### Risk Simulator

- Applicant Information
- Loan Information
- Credit Profile
- Probability of Default
- Credit Score
- Risk Band
- Lending Decision

---

### Macro Dashboard

Displays macroeconomic indicators including:

- Federal Funds Rate
- CPI
- Unemployment Rate

---

### Regulatory Assistant

Architecture prepared for future Retrieval-Augmented Generation (RAG) integration to answer regulatory and compliance questions.

---

## Project Structure

```
Credit-Card-Risk-Score-Builder
│
├── apps/
│   ├── app.py
│   ├── pages/
│   └── utils/
│
├── Data/
│   ├── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/jayeshhaslani/Credit-Card-Risk-Score-Builder.git
```

Move into the project

```bash
cd Credit-Card-Risk-Score-Builder
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run apps/app.py
```

---

## Future Improvements

- Gradient Boosting Scorecard
- XGBoost Comparison
- Probability Calibration
- SHAP Explainability
- Portfolio Stress Testing
- Live Economic Dashboard
- RBI Regulatory RAG Assistant
- Docker Deployment
- Cloud Deployment
- CI/CD Pipeline

---

## Learning Outcomes

Through this project I gained experience in:

- Credit Risk Modeling
- Statistical Scorecards
- Weight of Evidence Transformation
- Information Value Analysis
- Logistic Regression
- Feature Engineering
- Model Evaluation
- Streamlit Development
- API Integration
- Production Project Structuring

---

## Disclaimer

This project is intended for educational and portfolio purposes only.

It is **not** intended for use in production lending decisions.

---

## Author

**Jayesh Haslani**

GitHub: https://github.com/jayeshhaslani

LinkedIn: https://www.linkedin.com/in/jayesh-haslani-27a21a278/

---

## License

MIT License