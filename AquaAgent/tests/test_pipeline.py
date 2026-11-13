import os, sys, io, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings as S
from database import db
from tools import water_tools as wt
from agents.orchestrator import investigate
db.init(); DF, _ = wt.load_and_validate(S.DATA_PATH)

def test_validation_cleans_bad_rows():
    csv = "date,location,consumption,flow,night_flow,pressure,ph,turbidity\n2026-11-01,A,-5,1,1,50,7,2\n2026-11-02,A,100,1,1,50,7,2\n2026-11-02,A,100,1,1,50,7,2\nbad,A,100,1,1,50,7,2\n"
    df, issues = wt.load_and_validate(io.StringIO(csv)); assert len(df) == 1 and len(issues) >= 2
def test_missing_columns_raise():
    with pytest.raises(ValueError): wt.load_and_validate(io.StringIO("a,b\n1,2\n"))
def test_zone_b_anomaly(): a = wt.detect_anomaly(DF, "Zone B"); assert a["status"] == "HIGH" and 30 <= a["deviation_pct"] <= 40
def test_zone_a_normal(): assert wt.detect_anomaly(DF, "Zone A")["status"] == "NORMAL"
def test_leakage_high():
    l = wt.leakage_risk(DF, "Zone B", wt.detect_anomaly(DF, "Zone B")); assert l["level"] == "HIGH" and l["score"] >= 85
def test_quality_warning(): assert wt.check_quality(DF, "Zone C")["status"] == "WARNING"
def test_end_to_end_with_memory():
    r = investigate(DF, "Zone B"); assert r["recommendation"]["priority"] == "High" and len(r["history"]) >= 1 and "Inspect" in r["recommendation"]["action"]
    assert [s["agent"] for s in r["steps"]][:3] == ["Orchestrator", "Water Analysis Agent", "Leakage Agent"]
def test_normal_zone_skips_leakage():
    r = investigate(DF, "Zone A"); assert "leakage" not in r
