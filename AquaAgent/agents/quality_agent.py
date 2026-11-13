from agents.base_agent import BaseAgent


class QualityAgent(BaseAgent):
    name = "Water Quality Agent"
    role = "Monitor chemical and biological parameters (pH, turbidity) against safe water drinking standards."
    tool_names = ["check_water_quality"]
    required_tools = ["check_water_quality"]

    def offline_summary(self, obs, context):
        q = obs.get("check_water_quality", {})
        if "error" in q:
            return f"Water quality verification failed for {context['zone']}: {q['error']}"
        status = q.get("status", "OK")
        alerts = q.get("alerts", [])
        if status != "OK" or alerts:
            alert_details = "; ".join(f"{a['parameter']} at {a['mean']} (limit {a['allowed']})" for a in alerts)
            return f"⚠️ Water quality {status} in {context['zone']}: {alert_details}."
        return f"✅ Water quality in {context['zone']} is nominal (pH: {q.get('readings', {}).get('ph', 'N/A')}, Turbidity: {q.get('readings', {}).get('turbidity_ntu', 'N/A')} NTU)."
