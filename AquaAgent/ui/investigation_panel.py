"""Streamlit panel for the AI Investigation section. Day 3: call render() from app.py."""
import pandas as pd
import streamlit as st
from agents.orchestrator import Orchestrator
from tools import water_tools

BADGE = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢", "NONE": "🟢", "NORMAL": "🟢"}


def _show_report(r):
    if "error" in r:
        st.error(r["error"])
        return
    st.subheader(f"{BADGE.get(r['risk_level'], '⚪')} {r['zone']} — leakage risk {r['risk_level']} ({r['risk_score']}/100)")
    cols = st.columns(3)
    cols[0].metric("Consumption", f"{r['consumption'].get('current_l', 0):,.0f} L", f"{r['consumption'].get('deviation_pct', 0):+.1f}%")
    loss = (r.get("loss") or {}).get("potential_loss_l_per_day", 0)
    cols[1].metric("Potential wastage", f"{loss:,.0f} L/day")
    cols[2].metric("Priority", r["recommendation"]["priority"])
    st.markdown("**Agent workflow**")
    for w in r["workflow"]:
        icon = {"completed": "✅", "skipped": "⏭️", "error": "⚠️"}.get(w["status"], "•")
        st.write(f"{icon} {w['agent']} — {w['status']} ({w['tool_calls']} tool calls)")
    st.markdown("**Evidence**")
    for e in r["evidence"] or ["No abnormal indicators."]:
        st.write(f"- {e}")
    st.markdown("**AI recommendation**")
    for a in r["recommendation"]["actions"]:
        st.info(a)
    with st.expander("Agent activity log"):
        st.dataframe(pd.DataFrame(r["activity_log"]), use_container_width=True)
    st.download_button("Download report (.md)", r["report_markdown"], file_name=f"{r['zone']}_report.md")
    st.caption(f"Model: {r['llm']} · Risk scores are prototype indicators, not proof of a leak.")


def render():
    st.header("AI Investigation")
    zones = water_tools.list_zones()
    c1, c2 = st.columns([2, 1])
    zone = c1.selectbox("Zone", zones, index=zones.index("Zone B") if "Zone B" in zones else 0)
    question = st.text_input("Ask the agents (optional)", placeholder="Why is Zone B consumption high?")
    if c2.button("Investigate", type="primary"):
        with st.spinner("Orchestrator is coordinating agents..."):
            orch = Orchestrator()
            st.session_state["aqua_reports"] = (orch.ask(question) if question else
                                                {"reports": [orch.investigate(zone)], "normal_zones": []})
    if st.button("Scan all zones"):
        with st.spinner("Scanning..."):
            st.session_state["aqua_reports"] = Orchestrator().scan_all()
    out = st.session_state.get("aqua_reports")
    if out:
        for r in out["reports"]:
            _show_report(r)
        if out["normal_zones"]:
            st.success("Normal: " + ", ".join(out["normal_zones"]))
