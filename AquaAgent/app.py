"""
AquaAgent — AI-Powered Water Sustainability & Leakage Detection System
SDG 6: Clean Water and Sanitation
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings as S
from database import db
from tools import water_tools as wt
from tools import leakage_tools as lt
from tools import quality_tools as qt
from agents.orchestrator import Orchestrator

st.set_page_config(page_title="AquaAgent - SDG 6 Water Intelligence", page_icon="💧", layout="wide")
db.init_db()

# ─────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("💧 AquaAgent")
st.sidebar.caption("AI-Powered Water Sustainability · SDG 6")

upload = st.sidebar.file_uploader("Upload water data (CSV)", type="csv")
if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
        wt.set_dataframe(raw_df)
        source_name = upload.name
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        source_name = "default"
else:
    source_name = "Demo Dataset (90 days)"

df = wt.load_data()
zones = wt.list_zones()

st.sidebar.info(f"📂 **Source**: {source_name}\n\n📊 **Records**: {len(df):,} hourly readings")
selected_zones = st.sidebar.multiselect("Active Zones", zones, default=zones) or zones

# ─────────────────────────── Top KPIs & Title ────────────────────────────────
st.title("💧 AquaAgent — Intelligent Water Management System")
st.markdown("**AI Agent Collaboration & Leakage Detection** · Aligned with **UN SDG 6: Clean Water & Sanitation**")

# Calculate metrics across selected zones
zone_metrics = []
total_consumption = 0
total_wastage = 0
active_anomalies = 0
high_leakage_risks = 0
quality_warnings = 0

for z in selected_zones:
    c = wt.calculate_consumption(z)
    a = wt.detect_anomaly(z)
    l = lt.calculate_leakage_risk(z)
    q = qt.check_water_quality(z)
    w_loss = wt.estimate_water_loss(z)
    
    total_consumption += c.get("current_l", 0)
    loss_val = w_loss.get("potential_loss_l_per_day", 0)
    total_wastage += loss_val
    if a.get("status") in ("HIGH", "MEDIUM"):
        active_anomalies += 1
    if l.get("level") == "HIGH":
        high_leakage_risks += 1
    if q.get("status") != "OK":
        quality_warnings += 1

    zone_metrics.append({
        "Zone": z,
        "Current (L/day)": f"{c.get('current_l', 0):,.0f}",
        "Baseline (L/day)": f"{c.get('baseline_avg_l', 0):,.0f}",
        "Deviation": f"{a.get('deviation_pct', 0):+.1f}%",
        "Anomaly": a.get("status", "NORMAL"),
        "Leakage Risk": f"{l.get('level', 'LOW')} ({l.get('score', 0)}/100)",
        "Water Quality": q.get("status", "OK"),
        "Potential Wastage": f"{loss_val:,.0f} L/day"
    })

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💧 Daily Consumption", f"{total_consumption/1e3:,.1f}k L")
k2.metric("🚨 Consumption Alerts", active_anomalies)
k3.metric("🔴 High Leakage Risks", high_leakage_risks)
k4.metric("⚗️ Quality Warnings", quality_warnings)
k5.metric("📉 Est. Water Loss", f"{total_wastage:,.0f} L/day")

st.divider()

# ─────────────────────────── Tabs ────────────────────────────────────────────
tabs = st.tabs([
    "📊 System Overview",
    "🤖 Multi-Agent Investigation",
    "📈 Consumption Analytics",
    "🔴 Leakage Assessment",
    "⚗️ Water Quality",
    "🧠 Memory & Incident Log"
])

# ── 1. Overview ──────────────────────────────────────────────────────────────
with tabs[0]:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("📋 Real-Time Zone Status Matrix")
        st.dataframe(pd.DataFrame(zone_metrics), hide_index=True, use_container_width=True)
    with col_r:
        st.subheader("🌍 SDG 6 Sustainability Indicators")
        st.markdown("""
        - **Target 6.4:** Substantially increase water-use efficiency across all sectors.
        - **Wastage Reduction:** AI Agent early identification minimizes non-revenue water.
        """)
        st.metric("Potential Wastage Mitigated", f"{total_wastage:,.0f} L/day", delta="-Actionable", delta_color="inverse")
        st.caption("Heuristic estimation based on continuous nighttime deviation and pressure indicators.")

# ── 2. AI Investigation ──────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🤖 Autonomous Multi-Agent Investigation")
    st.caption("Watch specialized AI agents coordinate: Orchestrator ➔ Water Analysis ➔ Leakage Detection ➔ Recommendation.")

    c_z, c_q, c_b = st.columns([2, 3, 1])
    target_zone = c_z.selectbox("Target Zone for Investigation", zones, index=zones.index("Zone B") if "Zone B" in zones else 0)
    user_query = c_q.text_input("Investigative Prompt (optional)", placeholder=f"Why is {target_zone} consumption abnormal?")
    run_btn = c_b.button("🚀 Run Agents", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Orchestrator activating agents for {target_zone}..."):
            orch = Orchestrator()
            if user_query.strip():
                result = orch.ask(user_query)
                st.session_state["inv_report"] = result["reports"][0] if result["reports"] else orch.investigate(target_zone)
            else:
                st.session_state["inv_report"] = orch.investigate(target_zone)

    rep = st.session_state.get("inv_report")
    if rep:
        st.success(f"Investigation Complete for **{rep['zone']}** (Model: `{rep.get('llm', 'Offline Resilient Agent')}`)")
        
        # Step workflow sequence
        st.markdown("#### 🔄 Agent Workflow Execution")
        w_cols = st.columns(len(rep.get("workflow", [])))
        for idx, w in enumerate(rep.get("workflow", [])):
            with w_cols[idx]:
                status_icon = "✅" if w["status"] == "completed" else ("⏭️" if w["status"] == "skipped" else "⚠️")
                st.markdown(f"**{status_icon} {w['agent']}**")
                st.caption(f"Status: `{w['status']}` · Tools called: {w['tool_calls']}")
                st.info(w['summary'])

        st.divider()

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("#### 🔍 Discovered Evidence")
            for e in rep.get("evidence", []) or ["No abnormal indicators found."]:
                st.write(f"- {e}")
            
            leak_lvl = rep.get("risk_level", "LOW")
            score = rep.get("risk_score", 0)
            st.metric("Leakage Probability", f"{score}/100", f"Risk: {leak_lvl}")

        with res_col2:
            st.markdown("#### 💡 Actionable Recommendation")
            rec = rep.get("recommendation", {})
            prio = rec.get("priority", "Low")
            if prio == "High":
                st.error(f"### 🚨 Priority: {prio}")
            elif prio == "Medium":
                st.warning(f"### ⚠️ Priority: {prio}")
            else:
                st.info(f"### ℹ️ Priority: {prio}")

            for a in rec.get("actions", []):
                st.markdown(f"- **Action:** {a}")
            if rec.get("expected_benefit"):
                st.success(f"💧 **Expected Impact:** {rec['expected_benefit']}")

        with st.expander("⏱️ Detailed Orchestration & Tool Execution Log"):
            log_data = rep.get("activity_log", [])
            if log_data:
                st.dataframe(pd.DataFrame(log_data), hide_index=True, use_container_width=True)

        if rep.get("report_markdown"):
            st.download_button("📥 Download Official Audit Report (.md)", rep["report_markdown"], file_name=f"{rep['zone']}_investigation_report.md")

# ── 3. Consumption Analytics ─────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Water Consumption Trends")
    # Daily aggregation from loaded dataset
    plot_df = df[df["zone"].isin(selected_zones)].copy()
    plot_df["date"] = pd.to_datetime(plot_df["timestamp"]).dt.date
    daily_summary = plot_df.groupby(["date", "zone"])["consumption_l"].sum().reset_index()

    fig = px.line(daily_summary, x="date", y="consumption_l", color="zone", markers=True,
                  title="Daily Aggregate Consumption (L) Across Selected Zones",
                  labels={"consumption_l": "Litres/day", "date": "Date"})
    st.plotly_chart(fig, use_container_width=True)

    c_sel = st.selectbox("Select Zone for Detailed Hourly Flow", selected_zones, key="hourly_zone")
    z_hourly = df[df["zone"] == c_sel].tail(24 * 7)
    flow_col = "flow_lpm" if "flow_lpm" in df.columns else ("flow_lph" if "flow_lph" in df.columns else "consumption_l")
    fig_hourly = px.line(z_hourly, x="timestamp", y=flow_col,
                         title=f"{c_sel} - Last 7 Days Hourly Flow ({flow_col})")
    st.plotly_chart(fig_hourly, use_container_width=True)

# ── 4. Leakage Assessment ────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🔴 Heuristic Leakage Risk Breakdown")
    st.caption("Combines consumption anomaly (40%), continuous night flow (30%), pressure drop (20%), and persistence (10%).")
    for z in selected_zones:
        l_res = lt.calculate_leakage_risk(z)
        score = l_res["score"]
        lvl = l_res["level"]
        badge = "🔴" if lvl == "HIGH" else ("🟠" if lvl == "MEDIUM" else "🟢")
        with st.expander(f"{badge} {z} — Score: {score}/100 ({lvl} RISK)", expanded=(lvl == "HIGH")):
            st.progress(score / 100)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Evidence Log:**")
                if l_res["evidence"]:
                    for ev in l_res["evidence"]:
                        st.write(f"- {ev}")
                else:
                    st.write("No leakage indicators detected.")
            with c2:
                st.markdown("**Detailed Component Metrics:**")
                st.json(l_res["metrics"])

# ── 5. Water Quality ─────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("⚗️ Water Quality Parameter Tracking")
    q_rows = []
    for z in selected_zones:
        q_data = qt.check_water_quality(z)
        q_rows.append({
            "Zone": z,
            "Status": q_data.get("status"),
            "pH (Mean)": q_data["readings"].get("ph"),
            "Turbidity (NTU)": q_data["readings"].get("turbidity_ntu"),
            "Alerts": "; ".join(a["severity"] + " " + a["parameter"] for a in q_data.get("alerts", [])) or "None"
        })
    st.dataframe(pd.DataFrame(q_rows), hide_index=True, use_container_width=True)

    q_param = st.selectbox("Parameter View", ["ph", "turbidity_ntu"])
    q_plot = df[df["zone"].isin(selected_zones)].tail(24 * 7 * len(selected_zones))
    fig_q = px.line(q_plot, x="timestamp", y=q_param, color="zone", title=f"{q_param.upper()} Trends (Last 7 Days)")
    min_v, max_v = S.QUALITY_LIMITS[q_param]
    fig_q.add_hline(y=min_v, line_dash="dash", line_color="red", annotation_text=f"Min Safe ({min_v})")
    fig_q.add_hline(y=max_v, line_dash="dash", line_color="red", annotation_text=f"Max Safe ({max_v})")
    st.plotly_chart(fig_q, use_container_width=True)

# ── 6. Memory & Incidents ────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🧠 Long-Term Memory (Persistent Incident Store)")
    st.caption("Stores resolved and active incidents in SQLite to empower agent root-cause reasoning.")
    incidents = db.list_incidents(limit=20)
    if incidents:
        st.dataframe(pd.DataFrame(incidents), hide_index=True, use_container_width=True)
    else:
        st.info("No prior incidents recorded in memory.")
