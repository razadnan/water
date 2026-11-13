from agents.base_agent import BaseAgent


class WaterAnalysisAgent(BaseAgent):
    name = "Water Analysis Agent"
    role = "Measure current consumption against the historical baseline and check water quality."
    tool_names = ["calculate_consumption", "detect_anomaly", "check_water_quality"]
    required_tools = ["calculate_consumption", "detect_anomaly", "check_water_quality"]

    def offline_summary(self, obs, context):
        a, c, q = (obs.get(k, {}) for k in ("detect_anomaly", "calculate_consumption", "check_water_quality"))
        if "error" in a or "error" in c:
            return f"Could not analyse {context['zone']}: {a.get('error') or c.get('error')}"
        s = (f"{a['zone']} used {c['current_l']:,.0f} L on {c['latest_date']} versus a baseline of "
             f"{c['baseline_avg_l']:,.0f} L/day ({a['deviation_pct']:+.1f}%, z={a['z_score']}): "
             f"{a['status']} anomaly (confidence {a['confidence']}).")
        if q.get("alerts"):
            s += " Quality: " + "; ".join(f"{x['parameter']} mean {x['mean']} outside {x['allowed']}" for x in q["alerts"]) + "."
        return s
