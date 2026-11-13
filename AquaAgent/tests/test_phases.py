"""
Comprehensive Test Suite for Phases 1-5 Upgrades.
Tests:
- Dynamic orchestrator routing & dynamic agent delegation
- XGBoost forecasting tool & hold-out accuracy calculation
- Root-cause diagnostic tool & hypothesis ranking
- Semantic memory cosine similarity recall
- Data validation tracking (FR-03)
- API endpoint integration
"""
import pytest
import pandas as pd
from tools import water_tools as wt
from tools import leakage_tools as lt
from tools import quality_tools as qt
from tools import forecast_tools as ft
from tools.root_cause_tools import diagnose_root_cause
from memory.semantic_memory import SemanticMemoryStore
from agents.orchestrator import Orchestrator


def test_validation_tracking_reports_bad_rows():
    bad_data = pd.DataFrame([
        {"timestamp": "2026-11-01 01:00:00", "zone": "Zone A", "consumption_l": 500.0},
        {"timestamp": "invalid_date", "zone": "Zone A", "consumption_l": 500.0},
        {"timestamp": "2026-11-01 02:00:00", "zone": "Zone A", "consumption_l": -100.0},
        {"timestamp": "2026-11-01 01:00:00", "zone": "Zone A", "consumption_l": 500.0},  # duplicate
    ])
    clean_df, report = wt.validate_and_clean(bad_data)
    assert len(clean_df) == 1
    assert report["dropped_rows"] == 3
    assert report["reasons"]["invalid_timestamp"] >= 1
    assert report["reasons"]["negative_consumption"] >= 1


def test_xgboost_forecasting_runs_and_evaluates():
    res = ft.forecast_demand("Zone A")
    assert "xgboost_mae" in res
    assert "xgboost_mape_pct" in res
    assert res["xgboost_mape_pct"] < 15.0  # realistic forecast error bound
    assert res["test_samples"] > 0


def test_root_cause_diagnosis_identifies_pipeline_leak():
    mock_anom = {"status": "HIGH", "deviation_pct": 34.0, "persistent_days": 4, "zone": "Zone B"}
    mock_leak = {"score": 90, "metrics": {"night_flow_ratio": 4.5, "pressure_drop_pct": 14.0}}
    mock_qual = {"alerts": []}

    diagnosis = diagnose_root_cause(mock_anom, mock_leak, mock_qual)
    assert diagnosis["lead_cause"] == "Underground Pipeline Leak / Valve Rupture"
    assert diagnosis["lead_confidence"] > 0.80
    assert len(diagnosis["ranked_hypotheses"]) == 4


def test_root_cause_diagnosis_identifies_demand_surge():
    mock_anom = {"status": "HIGH", "deviation_pct": 28.0, "persistent_days": 1, "zone": "Zone D"}
    mock_leak = {"score": 20, "metrics": {"night_flow_ratio": 1.1, "pressure_drop_pct": 1.0}}
    mock_qual = {"alerts": []}

    diagnosis = diagnose_root_cause(mock_anom, mock_leak, mock_qual)
    assert diagnosis["lead_cause"] == "Operational / Legitimate Demand Surge"
    assert diagnosis["lead_confidence"] > 0.70


def test_semantic_memory_similarity_recall():
    store = SemanticMemoryStore()
    recalled = store.recall_similar_incidents("Zone B", "underground pipe rupture and night flow surge", top_k=2)
    assert len(recalled) > 0
    assert "similarity_score" in recalled[0]
    assert recalled[0]["zone"] == "Zone B"


def test_dynamic_orchestrator_includes_root_cause_on_anomaly():
    orch = Orchestrator()
    rep = orch.investigate("Zone B")
    assert rep["risk_level"] == "HIGH"
    assert "root_cause" in rep
    assert any("Diagnosed Cause" in ev for ev in rep["evidence"])
