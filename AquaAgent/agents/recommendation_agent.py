from agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    name = "Recommendation Agent"
    role = "Turn evidence and incident history into a prioritised, actionable recommendation."
    tool_names = ["get_historical_incidents"]
    required_tools = ["get_historical_incidents"]

    def _facts(self, obs, context):
        res = context.get("results", {})
        wa = res.get("water_analysis", {}).get("observations", {})
        lk = res.get("leakage", {}).get("observations", {})
        return (wa.get("detect_anomaly", {}), wa.get("check_water_quality", {}),
                lk.get("calculate_leakage_risk"), lk.get("estimate_water_loss"),
                obs.get("get_historical_incidents", {}).get("incidents", []))

    def structured(self, obs, context):
        zone = context["zone"]
        anom, qual, leak, loss, history = self._facts(obs, context)
        level = leak["level"] if leak and "error" not in leak else "NONE"
        actions, priority = [], "Low"
        if level == "HIGH":
            priority = "High"
            actions.append(f"Inspect the {zone} pipeline and associated valves; run a night-flow isolation test (01:00-04:00).")
        elif level == "MEDIUM":
            priority = "Medium"
            actions.append(f"Schedule a targeted inspection of {zone} and re-check night flow within 24 hours.")
        elif anom.get("status", "NORMAL") != "NORMAL":
            priority = "Medium"
            actions.append(f"Leakage evidence is weak in {zone}; check operational causes (new demand, scheduled activity, meter faults).")
        else:
            actions.append(f"No action required in {zone}; continue routine monitoring.")
        if qual.get("alerts"):
            sev = qual.get("status")
            priority = "High" if sev == "CRITICAL" else priority
            actions.append(f"Water-quality {sev}: verify sensor and take a manual sample in {zone} "
                           f"({', '.join(a['parameter'] for a in qual['alerts'])}).")
        similar = [h for h in history if "leak" in h["problem"].lower() and h["date"] != anom.get("date")]
        if level in ("HIGH", "MEDIUM") and similar:
            h = similar[0]
            actions.append(f"Similar incident on {h['date']} ({h['problem']}, resolved by: {h['resolution'] or 'n/a'}); check those components first.")
        benefit = None
        if loss and "error" not in loss and loss["potential_loss_l_per_day"] > 0:
            benefit = f"Up to {loss['potential_loss_l_per_day']:,.0f} L/day of potential wastage avoided."
        return {"priority": priority, "actions": actions, "expected_benefit": benefit,
                "similar_incident": similar[0] if (similar and level in ("HIGH", "MEDIUM")) else None}

    def offline_summary(self, obs, context):
        rec = self.structured(obs, context)
        anom, _, leak, _, _ = self._facts(obs, context)
        head = (f"{rec['priority']}-priority finding for {context['zone']}"
                f"{' (leakage risk ' + leak['level'] + ')' if leak else ''}. ")
        return head + " ".join(rec["actions"][:2])

    def post_process(self, result, context):
        rec = self.structured(result["observations"], context)
        rec["narrative"] = result["summary"]
        result["recommendation"] = rec
        return result
