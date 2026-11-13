# AquaAgent — AI-Powered Water Sustainability System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20IsolationForest-orange.svg)](https://xgboost.readthedocs.io/)
[![SDG 6](https://img.shields.io/badge/SDG%206-Clean%20Water-007ebd.svg)](https://sdgs.un.org/goals/goal6)
[![Tests Passing](https://img.shields.io/badge/tests-23%20passed-success.svg)](tests/)

**AquaAgent** is an intelligent multi-agent decision support system designed to monitor urban and municipal water infrastructure, detect abnormal consumption surges, investigate probable leakages with quantitative tool calling, and produce explainable, prioritized recommendations aligned with **UN SDG 6 (Clean Water and Sanitation)**.

---

## 📊 Benchmark Evaluation & Comparative Results

To avoid circular testing on trivial synthetic injections, AquaAgent is evaluated against a 60-day multi-zone distribution benchmark containing distinct operational scenarios:
- **Zone A**: Normal baseline throughout (false alarm evaluation)
- **Zone B**: Burst pipe rupture starting day 52 (+32% surge, 5.2× night flow, -15.2% pressure drop)
- **Zone C**: Incipient crack leak starting day 45 (+14% flow drift, chemical pH drop)
- **Zone D**: Legitimate high-demand community festival on days 30–31 (daytime surge, normal night flow, normal pressure)

### Detection Performance vs Baselines

```text
================== BENCHMARK EVALUATION RESULTS ==================
Model                              Precision   Recall     F1   False Alarm Rate
--------------------------------------------------------------------------------
AquaAgent (Multi-Signal Heuristic)     0.800    0.348  0.485              0.92%
Scikit-Learn IsolationForest (ML)      0.958    1.000  0.979              0.46%
Rolling-Median Threshold Baseline      0.667    0.087  0.154              0.46%
================================================================================
```

### Demand Forecasting Accuracy (XGBoost Regressor)
On a held-out 7-day test set (168 hourly time-steps):
- **XGBoost Mean Absolute Error (MAE)**: **92.6 L**
- **XGBoost Mean Absolute Percentage Error (MAPE)**: **4.04%**
- **Naive Weekly Persistence Baseline MAPE**: **4.83%**
- **Accuracy Improvement**: **+20.09% gain** over standard historical persistence.

---

## 🏗️ Multi-Agent Architecture

AquaAgent features 7 specialized cooperating agents:
1. **Orchestrator Agent**: Master task router with dynamic execution branching.
2. **Water Analysis Agent**: Baseline consumption deviation & Z-score measurement.
3. **Quality Agent**: Boundary monitoring for pH, turbidity, and chemical standards.
4. **Leakage Investigation Agent**: Minimum night flow (01:00–04:00) & pressure correlation.
5. **Demand Forecasting Agent**: Machine learning demand projections using XGBoost.
6. **Root-Cause Diagnostic Agent**: Quantitative hypothesis ranking (Leak vs Surge vs Sensor vs Quality).
7. **Recommendation & Action Agent**: Formulates prioritized action plans, dispatches work orders, and queries semantic memory.

```
                         ┌─────────────────────────┐
                         │   Orchestrator Agent    │
                         └───────────┬─────────────┘
                                     │ (Dynamic Routing)
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│  Water Analysis  │       │  Quality Agent   │       │ Demand Forecast Agent│
│      Agent       │       │ (pH, Turbidity)  │       │      (XGBoost)       │
└────────┬─────────┘       └────────┬─────────┘       └──────────┬───────────┘
         │                          │                            │
         ▼                          ▼                            ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│  Leakage Agent   │ ────► │ Root-Cause Agent │ ────► │ Recommendation Agent │
│(Night-flow ratio)│       │(Ranked Diagnosis)│       │  (Semantic Memory)   │
└──────────────────┘       └──────────────────┘       └──────────┬───────────┘
                                                                 │
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ Action/Dispatch Agent│
                                                      └──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/razadnan/water.git
cd water/AquaAgent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Benchmark & Run Evaluation
```bash
# 1. Generate realistic benchmark data + separate ground-truth labels:
python data/benchmark_generator.py

# 2. Run automated comparative evaluation against ML baselines:
python evaluate.py

# 3. Run full automated test suite:
pytest
```
*Expected: 23 passed, 1 skipped.*

### 3. Launch the Applications

#### Option A: Modern Command Center (Primary UI)
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)**.

#### Option B: Streamlit Dashboard (Secondary UI)
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)**.

#### Option C: Docker One-Command Run
```bash
docker-compose up --build
```

---

## 🤖 IBM watsonx.ai Integration (Optional)

AquaAgent features a **resilient hybrid execution architecture**:
- **Offline Mode (Default)**: Fully operational with deterministic rule-based narratives and tool validation. No API keys required.
- **Online Mode (IBM watsonx.ai)**: Enables IBM Granite foundation models (`ibm/granite-3-8b-instruct`) for natural language explanation generation and dynamic agent planning.

Configure credentials in `.env`:
```env
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_guid_here
WATSONX_MODEL_ID=ibm/granite-3-8b-instruct
```

---

## ⚠️ Known Limitations & Scope
- **Risk Score vs Ground Truth**: The leakage score (0–100) is a decision-support heuristic, not a physical confirmation of underground pipe rupture.
- **Actuation**: AquaAgent provides decision support and work-order dispatch. It does **not** directly control SCADA physical valves.
- **Data Scope**: The demo environment uses realistic physics-based distribution simulation. Results on specific municipal utility deployments will vary based on sensor density and pipe acoustics.
