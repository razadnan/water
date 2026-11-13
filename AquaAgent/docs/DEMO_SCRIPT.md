# 3–4 minute demo script
1. **Problem (20s):** Dashboards show "usage went up" but not why or what to do.
2. **Dashboard (30s):** Open app; show KPIs, trend chart, risk table. Zone B: HIGH, Zone C: quality warning (pH 5.8), Zone A normal.
3. **Agentic investigation (90s):** Select Zone B → *Run AI investigation*. Narrate each step: Orchestrator → Water Analysis (≈+35% vs baseline) → Leakage Agent (score 100/100: night flow 5.8x, pressure −15%, persistent) → Memory (Aug 12 Zone B leak) → Recommendation.
4. **Explainability (30s):** Point to evidence bullets and "rule-based / watsonx" label. Tools compute; LLM only phrases.
5. **Action + memory (20s):** Click *Save incident*; show it in Incident memory and the Agent activity log.
6. **SDG 6 (20s):** Potential wastage L/day, efficiency, water saved. State: supports efficiency, doesn't claim SDG attainment.
7. **Close (10s):** Future: IoT sensors, forecasting, valve control with approval.

Likely Q&A: *Why not just a threshold?* Multi-signal evidence + memory + explanation. *Is it a real leak?* It's a risk score guiding inspection. *Where's IBM?* watsonx.ai Granite for reasoning narrative; orchestration pattern maps to watsonx Orchestrate.
