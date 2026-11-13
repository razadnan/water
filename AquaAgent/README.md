# AquaAgent – AI-Powered Water Sustainability (SDG 6)
Multi-agent system that detects abnormal water usage, investigates probable leakage with tools + memory, and gives explainable recommendations.

## Run
```
pip install -r requirements.txt
python generate_data.py        # demo dataset (Zone B leak scenario)
python -m pytest -q            # 8 tests
streamlit run app.py
```
## IBM watsonx (optional)
Set env vars `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` (and optionally `WATSONX_MODEL`).
The Recommendation Agent then rewrites the evidence-based explanation with watsonx.ai (Granite). Without keys it falls back to the rule-based text, so the demo never breaks.
Numbers always come from tools; the LLM only phrases the result.

## Structure
app.py (dashboard) · agents/ (orchestrator, recommendation) · tools/ (validation, anomaly, leakage, quality, SDG) · memory/ + database/ (SQLite incidents, agent log) · tests/
