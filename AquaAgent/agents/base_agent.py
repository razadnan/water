"""Tool-using agent loop: LLM decides -> tool runs -> observation -> reasoning -> final."""
import json
import re
from tools.registry import call_tool, describe

PROTOCOL = (
    "Reply with ONE JSON object and nothing else.\n"
    'To call a tool: {"action":"tool","tool":"<tool name>","args":{"zone":"<zone>"}}\n'
    'When you have enough evidence: {"action":"final","summary":"<2-4 sentences using ONLY facts from tool observations>"}\n'
    "Never invent numbers. Call each tool at most once."
)


def extract_json(text):
    """Pull the first valid JSON object out of an LLM reply (handles ``` fences / chatter)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class BaseAgent:
    name = "Agent"
    role = ""
    tool_names = []        # tools this agent is allowed to use
    required_tools = []    # guardrail: always executed, even if the LLM forgets
    max_steps = 6

    def __init__(self, llm, log):
        self.llm, self.log = llm, log

    # -- to be overridden --------------------------------------------------
    def offline_summary(self, obs, context):
        return "Analysis complete."

    def post_process(self, result, context):
        return result

    # ----------------------------------------------------------------------
    def system_prompt(self):
        return (f"You are the {self.name} in AquaAgent, a water-sustainability system (SDG 6).\n"
                f"Role: {self.role}\nYou may use ONLY these tools:\n{describe(self.tool_names)}\n\n{PROTOCOL}")

    def _task(self, context):
        prior = {k: v.get("summary") for k, v in context.get("results", {}).items()}
        return json.dumps({"task": context.get("question") or f"Investigate {context['zone']}",
                           "zone": context["zone"], "findings_from_other_agents": prior}, default=str)

    def _run_tool(self, tool, zone, args=None, forced=False):
        args = dict(args or {})
        args["zone"] = zone
        result = call_tool(tool, **args)
        status = "error" if isinstance(result, dict) and "error" in result else "ok"
        self.log.add(self.name, f"tool:{tool}" + (" (guardrail)" if forced else ""), f"{status}")
        return result

    def run(self, context):
        zone = context["zone"]
        obs, summary, bad = {}, None, 0
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self._task(context)}]
        for _ in range(self.max_steps):
            hints = {"agent": self.name, "tools": self.tool_names, "observations": obs,
                     "default_args": {"zone": zone},
                     "summarize": lambda: self.offline_summary(obs, context)}
            raw = self.llm.chat(messages, hints)
            data = extract_json(raw)
            if not isinstance(data, dict):
                bad += 1
                self.log.add(self.name, "llm_parse_error", "reply was not valid JSON")
                if bad >= 2:
                    break
                messages += [{"role": "assistant", "content": str(raw)},
                             {"role": "user", "content": "Invalid. " + PROTOCOL}]
                continue
            if data.get("action") == "tool":
                tool = data.get("tool")
                if tool not in self.tool_names or tool in obs:
                    messages += [{"role": "assistant", "content": str(raw)},
                                 {"role": "user", "content": f"Tool '{tool}' is not allowed or already used. "
                                  f"Allowed unused: {[t for t in self.tool_names if t not in obs]}"}]
                    bad += 1
                    if bad >= 3:
                        break
                    continue
                obs[tool] = self._run_tool(tool, zone, data.get("args"))
                messages += [{"role": "assistant", "content": str(raw)},
                             {"role": "user", "content": f"Observation from {tool}: "
                              + json.dumps(obs[tool], default=str)[:3000]}]
            elif data.get("action") == "final":
                summary = data.get("summary")
                break
            else:
                bad += 1
                if bad >= 2:
                    break
        for tool in self.required_tools:      # guardrail: evidence must come from real tools
            if tool not in obs:
                obs[tool] = self._run_tool(tool, zone, forced=True)
        if not summary or not isinstance(summary, str):
            summary = self.offline_summary(obs, context)
            self.log.add(self.name, "summary_fallback", "used rule-based summary")
        result = {"agent": self.name, "observations": obs, "summary": summary,
                  "tool_calls": len(obs), "errors": [t for t, r in obs.items() if isinstance(r, dict) and "error" in r]}
        result = self.post_process(result, context)
        self.log.add(self.name, "completed", summary[:140])
        return result
