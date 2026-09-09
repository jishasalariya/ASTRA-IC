# ASTRA-IC: Semiconductor Reliability Screening System

Production-grade web application for predictive reliability screening of space-grade semiconductors during burn-in testing. Built with a clean light theme and a strict 2-stage ingestion workflow.

## 🚀 Quickstart

Run the dashboard with a single command:

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🔄 Strict 2-Stage Workflow

1. **Stage 1 (Landing Screen):**
   - No dataset is loaded by default.
   - Upload your own ATE CSV lot or click **"🚀 Load Sample Flight Batch (500 ICs)"** to test instantly.
2. **Stage 2 (Results Dashboard):**
   - **Section 1:** Executive KPI Summary (Total ICs, Static Pass Rate, Defect Catches, Testing Time Saved).
   - **Section 2:** Side-by-Side Visuals (Hour 0 Dynamic PAT scatter plot on left, 24h-168h Trajectory Forecaster on right).
   - **Section 3:** Single Component Inspector (`IC_0006` default) with plain English triage explanation and SHAP bar chart.
   - **Section 4:** Export Deliverables (Official PDF Compliance Report & CSV Data).
   - **Top Navigation:** Click **"⬅️ Upload Another Batch"** at any time to reset and return to Stage 1.

## 🧪 Verification Suite

Run the automated mathematical and pipeline verification test:

```powershell
python test_pipeline.py
```
