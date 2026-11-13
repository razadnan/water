from agents.base_agent import BaseAgent


class ActionAgent(BaseAgent):
    name = "Action / Dispatch Agent"
    role = "Translate validated recommendations into persistent incident tickets, dispatch work orders, and generate audit reports."
    tool_names = ["save_incident", "generate_report"]
    required_tools = ["generate_report"]

    def offline_summary(self, obs, context):
        rec_res = context.get("results", {}).get("recommendation", {})
        rec = rec_res.get("recommendation", {})
        actions = rec.get("actions", ["Routine maintenance."])
        prio = rec.get("priority", "Low")
        return f"Action Agent dispatched {prio}-priority work order for {context['zone']}: '{actions[0]}'. Incident report compiled."
