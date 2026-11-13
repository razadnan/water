from agents.base_agent import BaseAgent


class LeakageAgent(BaseAgent):
    name = "Leakage Investigation Agent"
    role = "Decide whether an anomaly looks like leakage using night flow, pressure and persistence evidence."
    tool_names = ["calculate_leakage_risk", "estimate_water_loss"]
    required_tools = ["calculate_leakage_risk", "estimate_water_loss"]

    def offline_summary(self, obs, context):
        r, loss = obs.get("calculate_leakage_risk", {}), obs.get("estimate_water_loss", {})
        if "error" in r:
            return f"Leakage assessment failed: {r['error']}"
        s = f"Leakage risk {r['level']} ({r['score']}/100). Evidence: " + ("; ".join(r["evidence"]) or "none") + "."
        if loss and "error" not in loss:
            s += f" Potential wastage about {loss['potential_loss_l_per_day']:,.0f} L/day."
        return s
