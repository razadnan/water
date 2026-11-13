"""Rule-based analysis pipeline. Day 2: the orchestrator will call these same tools."""
from tools.water_tools import validate_data, daily_consumption, consumption_summary, detect_anomaly, latest_status
from tools.leakage_tools import calculate_leakage_risk
from tools.quality_tools import check_water_quality

def build_alerts(latest, leaks, quality):
    alerts = []
    sev_map = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
    for r in latest.itertuples():
        if r.is_anomaly:
            alerts.append({"severity": sev_map[r.severity], "zone": r.zone,
                           "message": f"Consumption anomaly: {r.deviation_pct:+.1f}% vs baseline"})
    for l in leaks:
        if l["risk"] in ("HIGH", "MEDIUM"):
            alerts.append({"severity": "High" if l["risk"] == "HIGH" else "Medium", "zone": l["zone"],
                           "message": f"Possible leakage (risk score {l['score']}/100)"})
    for q in quality:
        if q["status"] != "OK":
            alerts.append({"severity": "Critical" if q["status"] == "CRITICAL" else "Medium",
                           "zone": q["zone"], "message": "; ".join(q["warnings"])})
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return sorted(alerts, key=lambda a: order[a["severity"]])

def run_analysis(raw_df):
    clean, issues = validate_data(raw_df)
    daily = daily_consumption(clean)
    anoms = detect_anomaly(daily)
    latest = latest_status(anoms)
    leaks = [calculate_leakage_risk(clean, anoms, z) for z in sorted(clean["zone"].unique())]
    quality = check_water_quality(clean)
    return {"clean": clean, "issues": issues, "daily": daily, "summary": consumption_summary(daily),
            "anoms": anoms, "latest": latest, "leaks": leaks, "quality": quality,
            "alerts": build_alerts(latest, leaks, quality),
            "potential_loss_l": sum(l["potential_loss_l"] for l in leaks)}
