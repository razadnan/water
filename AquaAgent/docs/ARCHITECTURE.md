# Architecture
```mermaid
flowchart TD
  U[Water Manager] --> UI[Streamlit Dashboard]
  UI -->|CSV upload / zone select| V[Validation Tool]
  V --> O[Orchestrator Agent]
  O --> WA[Water Analysis Agent<br/>detect_anomaly]
  WA -->|anomaly?| O
  O --> LA[Leakage Agent<br/>leakage_risk]
  O --> Q[Quality Tool<br/>check_quality]
  O --> M[(Memory: SQLite incidents)]
  LA --> RA[Recommendation Agent<br/>rules + watsonx.ai]
  M --> RA
  RA --> UI
  O --> LOG[(Agent activity log)]
  LOG --> UI
  UI -->|Save incident| M
```
# Flowchart
```mermaid
flowchart LR
  A[Upload data] --> B[Validate] --> C{Anomaly?}
  C -- No --> D[Show normal status]
  C -- Yes --> E[Leakage risk score] --> F[Recall past incidents] --> G[Explainable recommendation] --> H[Save incident / alert]
```
# Leakage risk score (prototype indicator, not proof of a leak)
Consumption anomaly +40 · night flow ≥3× +30 · pressure drop ≥10% +20 · persistence +10. HIGH ≥70, MEDIUM ≥40.
