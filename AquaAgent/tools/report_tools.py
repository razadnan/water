"""Markdown report generator."""


def generate_report(report):
    r = report
    rec = r.get("recommendation", {})
    lines = [f"# AquaAgent Investigation - {r.get('zone')}",
             f"**Date analysed:** {r.get('date')}  |  **Consumption status:** {r.get('status')}  |  "
             f"**Leakage risk:** {r.get('risk_level')} ({r.get('risk_score')}/100)", "", "## Evidence"]
    lines += [f"- {e}" for e in r.get("evidence", [])] or ["- No abnormal indicators found."]
    lines += ["", "## Recommendation", f"**Priority:** {rec.get('priority')}"]
    lines += [f"- {a}" for a in rec.get("actions", [])]
    if rec.get("expected_benefit"):
        lines.append(f"\n**Expected benefit:** {rec['expected_benefit']}")
    if rec.get("narrative"):
        lines += ["", "## AI Explanation", rec["narrative"]]
    lines += ["", "## Agent Workflow"]
    lines += [f"- {w['agent']}: {w['status']} ({w['tool_calls']} tool calls)" for w in r.get("workflow", [])]
    lines += ["", "_Risk scores are prototype indicators, not proof of a physical leak._"]
    return "\n".join(lines)
