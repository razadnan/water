from agents.base_agent import BaseAgent
from tools.root_cause_tools import diagnose_root_cause


class RootCauseAgent(BaseAgent):
    name = "Root-Cause Diagnostic Agent"
    role = "Evaluate and rank competing hypotheses (leak, demand surge, sensor drift, water contamination) using telemetry evidence."
    tool_names = ["calculate_consumption"]  # uses upstream context
    required_tools = []

    def run(self, context):
        res = context.get("results", {})
        wa = res.get("water_analysis", {}).get("observations", {})
        lk = res.get("leakage", {}).get("observations", {})
        qa = res.get("quality", {}).get("observations", {})

        anom = wa.get("detect_anomaly", {"status": "NORMAL", "deviation_pct": 0.0, "persistent_days": 0, "zone": context["zone"]})
        leak = lk.get("calculate_leakage_risk", {"score": 0, "metrics": {}})
        qual = qa.get("check_water_quality", {"alerts": []})

        diagnosis = diagnose_root_cause(anom, leak, qual)
        summary = f"Root-cause diagnostic: {diagnosis['lead_cause']} ({diagnosis['lead_confidence']*100:.0f}% confidence). {diagnosis['explanation']}"
        self.log.add(self.name, "diagnosed", diagnosis["lead_cause"])

        return {
            "agent": self.name,
            "observations": {"diagnose_root_cause": diagnosis},
            "summary": summary,
            "tool_calls": 1,
            "errors": []
        }

    def offline_summary(self, obs, context):
        return obs.get("diagnose_root_cause", {}).get("explanation", "Diagnostic complete.")
