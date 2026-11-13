"""Orchestrator: plans the investigation, runs sub-agents, branches on evidence, acts, reports."""
import json
import re
from agents.activity_log import ActivityLog
from agents.base_agent import extract_json
from agents.water_agent import WaterAnalysisAgent
from agents.leakage_agent import LeakageAgent
from agents.recommendation_agent import RecommendationAgent
from llm.llm_client import get_llm
from memory.memory import ShortTermMemory
from tools import water_tools
from tools.registry import call_tool

ORDER = ["water_analysis", "leakage", "recommendation"]


class Orchestrator:
    def __init__(self, llm=None, log=None):
        self.llm = llm or get_llm()
        self.log = log or ActivityLog()
        self.agents = {"water_analysis": WaterAnalysisAgent(self.llm, self.log),
                       "leakage": LeakageAgent(self.llm, self.log),
                       "recommendation": RecommendationAgent(self.llm, self.log)}

    # ---- planning ------------------------------------------------------
    def _plan(self, zone, question):
        system = ("You are the Orchestrator of AquaAgent. Choose which specialist agents to run, in order.\n"
                  "Available: water_analysis (consumption+quality), leakage (leakage risk), "
                  "recommendation (actions + incident memory).\n"
                  'Reply ONLY with JSON: {"plan":["water_analysis","leakage","recommendation"],"reason":"..."}')
        user = json.dumps({"zone": zone, "request": question or f"Investigate abnormal water usage in {zone}"})
        raw = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                            {"mode": "plan", "default_plan": ORDER})
        data = extract_json(raw) or {}
        plan = [p for p in dict.fromkeys(data.get("plan") or []) if p in self.agents]
        if not plan:                              # unusable LLM plan -> safe default
            self.log.add("Orchestrator", "plan_fallback", "LLM plan invalid; using default plan")
            return list(ORDER), "default plan (LLM plan unusable)"
        if "water_analysis" not in plan:          # evidence must come first
            plan.insert(0, "water_analysis")
        if "recommendation" not in plan:          # every investigation ends with a decision
            plan.append("recommendation")
        plan = [p for p in ORDER if p in plan]
        return plan, str(data.get("reason", "default plan"))

    @staticmethod
    def _needs_leakage(ctx):
        wa = ctx["results"].get("water_analysis")
        if not wa:
            return True
        return wa["observations"].get("detect_anomaly", {}).get("status", "NORMAL") != "NORMAL"

    # ---- main entry points ----------------------------------------------
    def investigate(self, zone, question=None, trigger="manual"):
        canon = water_tools.resolve_zone(zone)
        if canon is None:
            return {"error": f"Unknown zone '{zone}'. Available: {water_tools.list_zones()}"}
        zone = canon
        stm = ShortTermMemory()
        stm.set("zone", zone)
        log_start = len(self.log.events)
        self.log.add("Orchestrator", "started", f"{zone} (trigger: {trigger})")
        plan, reason = self._plan(zone, question)
        self.log.add("Orchestrator", "plan", " -> ".join(plan))
        ctx = {"zone": zone, "question": question, "results": {}}
        workflow = []
        for step in list(plan):
            agent = self.agents[step]
            if step == "recommendation" and "leakage" not in plan and "leakage" not in ctx["results"] \
                    and self._needs_leakage(ctx):
                self.log.add("Orchestrator", "override", "anomaly found -> adding leakage agent to plan")
                plan.insert(plan.index("recommendation"), "leakage")
                lk = self.agents["leakage"]
                self.log.add("Orchestrator", "delegating", lk.name)
                res = lk.run(ctx)
                ctx["results"]["leakage"] = res
                stm.set("leakage", res["summary"])
                workflow.append({"agent": lk.name, "status": "error" if res["errors"] else "completed",
                                 "tool_calls": res["tool_calls"], "summary": res["summary"]})
            if step == "leakage" and not self._needs_leakage(ctx):
                self.log.add("Orchestrator", "skipped leakage agent", "no consumption anomaly detected")
                workflow.append({"agent": agent.name, "status": "skipped", "tool_calls": 0, "summary": "No anomaly - not needed."})
                continue
            self.log.add("Orchestrator", "delegating", agent.name)
            res = agent.run(ctx)
            ctx["results"][step] = res
            stm.set(step, res["summary"])
            workflow.append({"agent": agent.name, "status": "error" if res["errors"] else "completed",
                             "tool_calls": res["tool_calls"], "summary": res["summary"]})
        report = self._assemble(ctx, plan, reason, workflow)
        self._act(report)
        self.log.add("Orchestrator", "finished", f"{zone}: leakage risk {report['risk_level']}")
        report["activity_log"] = self.log.rows()[log_start:]
        report["llm"] = self.llm.name
        return report

    def scan_all(self):
        """Check every zone; run the full agent workflow only where something is abnormal."""
        reports, ok = [], []
        for z in water_tools.list_zones():
            a = call_tool("detect_anomaly", zone=z)
            q = call_tool("check_water_quality", zone=z)
            if a.get("status", "NORMAL") != "NORMAL" or q.get("alerts"):
                reports.append(self.investigate(z, trigger="scheduled scan"))
            else:
                ok.append(z)
        return {"reports": reports, "normal_zones": ok}

    def ask(self, question):
        m = re.search(r"zone\s*([a-z0-9]+)", question, re.I)
        if m:
            return {"reports": [self.investigate(f"Zone {m.group(1)}", question=question, trigger="user question")],
                    "normal_zones": []}
        return self.scan_all()

    # ---- assembly --------------------------------------------------------
    def _assemble(self, ctx, plan, reason, workflow):
        res = ctx["results"]
        wa = res.get("water_analysis", {}).get("observations", {})
        lk = res.get("leakage", {}).get("observations", {})
        rec_res = res.get("recommendation", {})
        anom, cons, qual = wa.get("detect_anomaly", {}), wa.get("calculate_consumption", {}), wa.get("check_water_quality", {})
        leak, loss = lk.get("calculate_leakage_risk"), lk.get("estimate_water_loss")
        rec = rec_res.get("recommendation", {"priority": "Low", "actions": [], "narrative": ""})
        evidence = []
        if anom and "error" not in anom and cons and "error" not in cons:
            evidence.append(f"Consumption {cons['current_l']:,.0f} L vs baseline {cons['baseline_avg_l']:,.0f} L/day "
                            f"({anom['deviation_pct']:+.1f}%, z={anom['z_score']})")
        if leak and "error" not in leak:
            evidence += [e for e in leak["evidence"] if not e.startswith(("+", "-")) or "consumption" not in e][:]
        if rec.get("similar_incident"):
            h = rec["similar_incident"]
            evidence.append(f"Similar previous incident {h['date']}: {h['problem']} -> {h['resolution'] or 'n/a'}")
        for a in qual.get("alerts", []) if isinstance(qual, dict) else []:
            evidence.append(f"{a['parameter']} mean {a['mean']} outside allowed {a['allowed']} ({a['severity']})")
        return {"zone": ctx["zone"], "date": anom.get("date"), "status": anom.get("status", "UNKNOWN"),
                "anomaly": anom, "consumption": cons, "leakage": leak, "loss": loss, "quality": qual,
                "risk_score": leak["score"] if leak and "error" not in leak else 0,
                "risk_level": leak["level"] if leak and "error" not in leak else "NONE",
                "evidence": evidence, "recommendation": rec,
                "history": rec_res.get("observations", {}).get("get_historical_incidents", {}).get("incidents", []),
                "plan": plan, "plan_reason": reason, "workflow": workflow, "incident_saved": False}

    def _act(self, report):
        """Action step: persist incidents (MEDIUM/HIGH risk) and produce the report."""
        if report["risk_level"] in ("HIGH", "MEDIUM") and report.get("date"):
            out = call_tool("save_incident", zone=report["zone"], problem="Suspected leakage / abnormal consumption",
                            risk=report["risk_level"], recommendation=" ".join(report["recommendation"].get("actions", []))[:300],
                            date=report["date"])
            report["incident_saved"] = bool(out.get("saved"))
            self.log.add("Orchestrator", "incident", "saved to memory" if out.get("saved") else out.get("reason", "not saved"))
        report["report_markdown"] = call_tool("generate_report", report=report)
