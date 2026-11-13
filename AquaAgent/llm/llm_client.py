"""LLM layer: IBM watsonx.ai (primary) with a deterministic offline fallback.

Every agent talks to the LLM through `chat(messages, hints)`. `hints` is only used by the
offline model so the whole workflow still runs (and can be tested) without credentials.
"""
import json
from config import settings


class OfflineLLM:
    name = "offline-rule-based"

    def chat(self, messages, hints=None):
        hints = hints or {}
        if hints.get("mode") == "plan":
            return json.dumps({"plan": hints.get("default_plan", []),
                               "reason": "Standard investigation order: analyse, assess leakage, recommend."})
        obs = hints.get("observations", {})
        for tool in hints.get("tools", []):
            if tool not in obs:
                return json.dumps({"action": "tool", "tool": tool, "args": hints.get("default_args", {})})
        summarize = hints.get("summarize")
        return json.dumps({"action": "final", "summary": summarize() if summarize else "Done."})


class WatsonxLLM:
    def __init__(self):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        try:
            from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
            params = TextChatParameters(temperature=0.0, max_tokens=600)
        except ImportError:
            params = {"temperature": 0.0, "max_tokens": 600}
        self.name = f"watsonx.ai:{settings.WATSONX_MODEL_ID}"
        self.model = ModelInference(
            model_id=settings.WATSONX_MODEL_ID,
            credentials=Credentials(url=settings.WATSONX_URL, api_key=settings.WATSONX_API_KEY),
            project_id=settings.WATSONX_PROJECT_ID, params=params)

    def chat(self, messages, hints=None):
        resp = self.model.chat(messages=messages)
        return resp["choices"][0]["message"]["content"]


class ResilientLLM:
    """Uses the primary model; on any failure transparently falls back to offline logic."""

    def __init__(self, primary=None):
        self.primary = primary
        self.offline = OfflineLLM()
        self.fallback_events = 0

    @property
    def name(self):
        if self.primary is None:
            return self.offline.name
        return self.primary.name + (f" (+{self.fallback_events} offline fallbacks)" if self.fallback_events else "")

    def chat(self, messages, hints=None):
        if self.primary is not None:
            try:
                return self.primary.chat(messages, hints)
            except Exception:
                self.fallback_events += 1
        return self.offline.chat(messages, hints)


def get_llm(force_offline=None):
    offline = settings.FORCE_OFFLINE if force_offline is None else force_offline
    if offline or not (settings.WATSONX_API_KEY and settings.WATSONX_PROJECT_ID):
        return ResilientLLM(None)
    try:
        return ResilientLLM(WatsonxLLM())
    except Exception:  # SDK missing / bad credentials -> still run
        return ResilientLLM(None)
