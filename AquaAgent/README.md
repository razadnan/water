# AquaAgent — AI-Powered Water Sustainability System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![SDG 6](https://img.shields.io/badge/SDG%206-Clean%20Water-007ebd.svg)](https://sdgs.un.org/goals/goal6)
[![Tests Passing](https://img.shields.io/badge/tests-17%20passed-success.svg)](tests/)

**AquaAgent** is an intelligent multi-agent decision support system designed to monitor urban and municipal water infrastructure, detect abnormal consumption surges, investigate probable leakages with tool calling and persistent memory, and produce explainable, prioritized recommendations aligned with **UN SDG 6 (Clean Water and Sanitation)**.

---

## 🏗️ Architecture

AquaAgent implements a strict separation of concerns:
- **LLM = Reasoning & Orchestration** (Agent loop with schema validation and task routing)
- **Tools = Deterministic Computation** (Z-scores, night-flow ratios, pressure drops, statistical validation)
- **Memory = Incident Recall & Context** (SQLite store for previous incidents and resolution history)

```
                       ┌─────────────────────────┐
                       │  Orchestrator Agent     │
                       └───────────┬─────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Water Analysis  │     │     Leakage      │     │    Recommendation    │
│      Agent       │     │   Investigation  │     │        Agent         │
└────────┬─────────┘     └────────┬─────────┘     └──────────┬───────────┘
         │                        │                          │
         ▼                        ▼                          ▼
 ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
 │  Data Tools  │         │ Risk Scorer  │         │ Long-Term Memory │
 └──────────────┘         └──────────────┘         └──────────────────┘
```

> **Note on Risk Scores**: The leakage score (0–100) is a deterministic heuristic indicator combining consumption surge, abnormal nighttime flow (01:00–04:00), and pressure drop. It is decision-support evidence, not physical proof of an underground pipe rupture.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/razadnan/water.git
cd water/AquaAgent

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Simulated Telemetry
Generate the 30-day baseline dataset with injected leak patterns:
```bash
python data/generate_data.py
```

### 3. Run Automated Tests
Execute the full test suite with clean test discovery:
```bash
pytest
```
*Expected: 17 passed, 1 skipped (live watsonx credentials skipped if unconfigured).*

### 4. Launch the Applications

#### Option A: Modern Command Center (Primary UI)
Run the FastAPI production interface with live telemetry, chat assistant, and multi-agent workflow:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

#### Option B: Streamlit Dashboard (Secondary UI)
```bash
streamlit run app.py
```
Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 🤖 IBM watsonx.ai Integration (Optional)

AquaAgent features a **resilient hybrid execution architecture**:
- **Offline Mode (Default)**: Fully operational with deterministic rule-based narratives and tool validation. No API keys required.
- **Online Mode (IBM watsonx.ai)**: Enables IBM Granite foundation models (`ibm/granite-3-8b-instruct`) for natural language explanation generation and dynamic agent planning.

To enable IBM watsonx.ai, create a `.env` file in `AquaAgent/`:
```env
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_guid_here
WATSONX_MODEL_ID=ibm/granite-3-8b-instruct
```

---

## 📂 Project Structure

```
AquaAgent/
├── agents/                  # Specialized AI agents
│   ├── base_agent.py        # Agent loop with required_tools guardrail
│   ├── orchestrator.py      # Master workflow coordinator
│   ├── water_agent.py       # Consumption baseline & quality analysis
│   ├── leakage_agent.py     # Night-flow & pressure correlation
│   └── recommendation_agent.py # Action plan generation
├── tools/                   # Pure Python calculation tools (no LLM dependencies)
│   ├── water_tools.py       # Consumption metrics & anomaly detection
│   ├── leakage_tools.py     # Night-flow heuristic scoring
│   ├── quality_tools.py     # pH & turbidity boundary checks
│   └── registry.py          # Unified tool dispatcher
├── models/                  # ML & statistical anomaly models
│   └── anomaly_model.py     # Z-score & rolling baseline wrappers
├── data/
│   ├── generate_data.py     # Telemetry generator
│   └── water_data.csv       # 30-day hourly synthetic dataset
├── database/                # SQLite persistent storage
│   └── db.py                # Incident history & activity logs
├── memory/                  # Agent memory abstraction
│   └── memory.py            # Short-term (context) & Long-term (DB)
├── static/                  # Web assets
│   └── index.html           # Modern Command Center single-page app
├── server.py                # FastAPI application & REST endpoints
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Production dependencies
└── tests/                   # Automated pytest suite
```

---

## 🌍 UN Sustainable Development Goal 6 Alignment

AquaAgent contributes to **SDG Target 6.4**:
> *"By 2030, substantially increase water-use efficiency across all sectors and ensure sustainable withdrawals and supply of freshwater."*

By correlating nighttime minimum flow rates with localized pressure drops, AquaAgent flags potential non-revenue water (NRW) losses early, allowing utilities to isolate distribution zones before catastrophic pipe failures occur.
