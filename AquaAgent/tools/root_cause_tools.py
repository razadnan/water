"""
Root-Cause Analysis Tool for Water Anomalies.
Evaluates hypotheses against quantitative evidence and historical signals.
Hypotheses evaluated:
1. PIPELINE_LEAKAGE (High night flow, pressure drop, persistence)
2. PLANNED_DEMAND_SURGE (High daytime flow, normal night flow, normal pressure)
3. SENSOR_OR_METER_FAULT (Negative flow, zero variance, abrupt drop/jump)
4. WATER_QUALITY_EVENT (pH or turbidity violation without mechanical flow change)
"""
from typing import Dict, Any


def diagnose_root_cause(anomaly_data: Dict[str, Any], leakage_data: Dict[str, Any], quality_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ranks potential root causes based on deterministic heuristic scoring of physical telemetry.
    Returns ranked hypotheses with confidence scores and evidence breakdown.
    """
    hypotheses = []

    # Telemetry extractions
    is_anomaly = anomaly_data.get("status") in ("HIGH", "MEDIUM")
    dev_pct = anomaly_data.get("deviation_pct", 0)
    leak_score = leakage_data.get("score", 0)
    leak_metrics = leakage_data.get("metrics", {})
    night_ratio = leak_metrics.get("night_flow_ratio", 1.0)
    pressure_drop = leak_metrics.get("pressure_drop_pct", 0.0)
    persist_days = anomaly_data.get("persistent_days", 0)
    quality_alerts = quality_data.get("alerts", [])

    # 1. Evaluate Pipeline Leakage Hypothesis
    leak_prob = 0.0
    leak_evidence = []
    if night_ratio >= 3.0:
        leak_prob += 0.40
        leak_evidence.append(f"Continuous nighttime minimum flow ({night_ratio:.1f}x baseline)")
    elif night_ratio >= 1.5:
        leak_prob += 0.20
        leak_evidence.append(f"Elevated nighttime minimum flow ({night_ratio:.1f}x baseline)")

    if pressure_drop >= 10.0:
        leak_prob += 0.35
        leak_evidence.append(f"Hydraulic pressure decrease ({pressure_drop:.1f}% reduction)")
    elif pressure_drop >= 4.0:
        leak_prob += 0.15
        leak_evidence.append(f"Moderate localized pressure drop ({pressure_drop:.1f}%)")

    if persist_days >= 3:
        leak_prob += 0.25
        leak_evidence.append(f"Persisted anomaly across {persist_days} consecutive days")

    leak_conf = min(0.99, max(0.05, leak_prob))
    hypotheses.append({
        "cause": "Underground Pipeline Leak / Valve Rupture",
        "category": "PHYSICAL_INFRASTRUCTURE",
        "confidence": round(leak_conf, 2),
        "primary_evidence": leak_evidence,
        "is_lead_cause": False
    })

    # 2. Evaluate Legitimate Demand Surge (e.g. Festival, Industrial Shift)
    surge_prob = 0.0
    surge_evidence = []
    if is_anomaly and dev_pct > 15.0:
        if night_ratio < 1.3:
            surge_prob += 0.50
            surge_evidence.append("Night flow remains nominal (< 1.3x baseline) despite high daily consumption")
        if pressure_drop < 3.0:
            surge_prob += 0.40
            surge_evidence.append("Distribution network pressure maintained within normal limits")
        if persist_days <= 2:
            surge_prob += 0.10
            surge_evidence.append("Transient event duration (< 3 days)")

    surge_conf = min(0.95, max(0.02, surge_prob))
    hypotheses.append({
        "cause": "Operational / Legitimate Demand Surge",
        "category": "CONSUMPTION_BEHAVIOR",
        "confidence": round(surge_conf, 2),
        "primary_evidence": surge_evidence,
        "is_lead_cause": False
    })

    # 3. Evaluate Sensor / Telemetry Malfunction
    sensor_prob = 0.0
    sensor_evidence = []
    if dev_pct < -50.0:
        sensor_prob += 0.70
        sensor_evidence.append("Sudden complete flow loss without documented pump shutdown")
    if anomaly_data.get("z_score", 0) > 15.0 and night_ratio < 1.0:
        sensor_prob += 0.50
        sensor_evidence.append("Unphysical single-hour spike inconsistent with hydraulic capacity")

    sensor_conf = min(0.90, max(0.01, sensor_prob))
    hypotheses.append({
        "cause": "Flow Meter or Telemetry Sensor Malfunction",
        "category": "INSTRUMENTATION",
        "confidence": round(sensor_conf, 2),
        "primary_evidence": sensor_evidence or ["Telemetry consistent with physical flow readings"],
        "is_lead_cause": False
    })

    # 4. Evaluate Contamination / Water Quality Anomaly
    quality_prob = 0.0
    quality_evidence = []
    if quality_alerts:
        quality_prob = 0.85
        quality_evidence = [f"{a['parameter']} reading ({a['mean']}) breached safe limit ({a['allowed']})" for a in quality_alerts]
    hypotheses.append({
        "cause": "Water Quality / Chemical Contamination Event",
        "category": "ENVIRONMENTAL_SAFETY",
        "confidence": round(quality_prob, 2),
        "primary_evidence": quality_evidence or ["Water quality parameters nominal"],
        "is_lead_cause": False
    })

    # Sort hypotheses by confidence descending
    hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
    if hypotheses:
        hypotheses[0]["is_lead_cause"] = True

    lead = hypotheses[0]
    return {
        "zone": anomaly_data.get("zone", "Unknown"),
        "lead_cause": lead["cause"],
        "lead_confidence": lead["confidence"],
        "explanation": f"Lead diagnostic: {lead['cause']} ({lead['confidence']*100:.0f}% confidence). Key factor: {'; '.join(lead['primary_evidence'][:2])}",
        "ranked_hypotheses": hypotheses
    }
