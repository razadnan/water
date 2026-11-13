import json
from agents.base_agent import extract_json
from agents.orchestrator import Orchestrator
from llm.llm_client import OfflineLLM, ResilientLLM
from tools import water_tools as w, leakage_tools as l, quality_tools as q
from tools.registry import call_tool
from database import db


# ---------------- tools ----------------
def test_zone_b_high_anomaly_and_leakage():
    assert w.detect_anomaly("Zone B")["status"] == "HIGH"
    r = l.calculate_leakage_risk("Zone B")
    assert r["level"] == "HIGH" and r["metrics"]["night_flow_ratio"] >= 3


def test_zone_a_normal():
    assert w.detect_anomaly("Zone A")["status"] == "NORMAL"
    assert l.calculate_leakage_risk("Zone A")["level"] == "LOW"


def test_zone_c_quality_alert():
    assert q.check_water_quality("Zone C")["status"] == "CRITICAL"
    assert q.check_water_quality("Zone A")["status"] == "OK"


def test_unknown_zone_returns_error_not_exception():
    assert "error" in w.detect_anomaly("Zone Z")
    assert "error" in call_tool("no_such_tool")


def test_water_loss_positive_for_leak():
    assert w.estimate_water_loss("Zone B")["potential_loss_l_per_day"] > 10000


# ---------------- helpers ----------------
def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("no json here") is None


# ---------------- agents / orchestrator ----------------
def test_full_workflow_zone_b():
    rep = Orchestrator(llm=ResilientLLM(None)).investigate("zone b")
    assert rep["risk_level"] == "HIGH" and rep["recommendation"]["priority"] == "High"
    assert rep["plan"] == ["water_analysis", "leakage", "recommendation"]
    assert all(w_["status"] == "completed" for w_ in rep["workflow"])
    assert any("Similar previous incident" in e for e in rep["evidence"])
    assert rep["incident_saved"] is True
    assert "Inspect" in rep["recommendation"]["actions"][0]


def test_agents_really_call_tools():
    rep = Orchestrator(llm=ResilientLLM(None)).investigate("Zone B")
    tools_called = {e["event"] for e in rep["activity_log"] if e["event"].startswith("tool:")}
    assert {"tool:detect_anomaly", "tool:calculate_leakage_risk", "tool:get_historical_incidents"} <= tools_called


def test_orchestrator_skips_leakage_for_normal_zone():
    rep = Orchestrator(llm=ResilientLLM(None)).investigate("Zone A")
    leak_step = [x for x in rep["workflow"] if x["agent"].startswith("Leakage")][0]
    assert leak_step["status"] == "skipped"
    assert rep["risk_level"] == "NONE" and "No action required" in rep["recommendation"]["actions"][0]
    assert rep["incident_saved"] is False


def test_duplicate_incident_not_saved_twice():
    o = Orchestrator(llm=ResilientLLM(None))
    o.investigate("Zone B")
    assert o.investigate("Zone B")["incident_saved"] is False
    assert len([i for i in db.list_incidents("Zone B") if i["status"] == "Open"]) == 1


def test_scan_all_only_investigates_abnormal_zones():
    out = Orchestrator(llm=ResilientLLM(None)).scan_all()
    assert {r["zone"] for r in out["reports"]} == {"Zone B", "Zone C"}
    assert out["normal_zones"] == ["Zone A"]


def test_ask_extracts_zone_from_question():
    out = Orchestrator(llm=ResilientLLM(None)).ask("Why is Zone B consumption high?")
    assert out["reports"][0]["zone"] == "Zone B"


def test_zone_c_quality_critical_raises_priority():
    rep = Orchestrator(llm=ResilientLLM(None)).investigate("Zone C")
    assert rep["recommendation"]["priority"] == "High"
    assert any("Water-quality CRITICAL" in a for a in rep["recommendation"]["actions"])


# ---------------- resilience ----------------
class GarbageLLM:
    name = "garbage"
    def chat(self, messages, hints=None):
        return "I think there might be a leak, who knows!"


class ForgetfulLLM:
    """Jumps to 'final' immediately without calling tools, with a made-up claim."""
    name = "forgetful"
    def chat(self, messages, hints=None):
        if (hints or {}).get("mode") == "plan":
            return json.dumps({"plan": ["recommendation"]})   # tries to skip analysis
        return json.dumps({"action": "final", "summary": "All fine."})


class CrashingLLM:
    name = "crash"
    def chat(self, messages, hints=None):
        raise RuntimeError("watsonx unavailable")


def test_garbage_llm_still_produces_tool_based_result():
    rep = Orchestrator(llm=GarbageLLM()).investigate("Zone B")
    assert rep["risk_level"] == "HIGH"          # evidence came from tools, not from the LLM text


def test_forgetful_llm_cannot_skip_tools_or_analysis():
    rep = Orchestrator(llm=ForgetfulLLM()).investigate("Zone B")
    assert rep["plan"][0] == "water_analysis" and rep["risk_level"] == "HIGH"


def test_llm_outage_falls_back_to_offline():
    llm = ResilientLLM(CrashingLLM())
    rep = Orchestrator(llm=llm).investigate("Zone B")
    assert rep["risk_level"] == "HIGH" and llm.fallback_events > 0


def test_scan_reports_have_separate_activity_logs():
    out = Orchestrator(llm=ResilientLLM(None)).scan_all()
    for r in out["reports"]:
        assert all(r["zone"] in e["detail"] or e["agent"] != "Orchestrator" or e["event"] not in ("started",)
                   for e in r["activity_log"])
        assert sum(1 for e in r["activity_log"] if e["event"] == "started") == 1
