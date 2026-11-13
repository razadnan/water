"""Leakage-risk scoring (prototype score, NOT proof of a physical leak)."""
from config import settings
from tools.water_tools import _split, detect_anomaly


def calculate_leakage_risk(zone):
    ctx, err = _split(zone)
    if err:
        return err
    anom = detect_anomaly(zone)
    hours = list(settings.NIGHT_HOURS)
    base_n = float(ctx["base_hourly"][ctx["base_hourly"]["timestamp"].dt.hour.isin(hours)]["consumption_l"].mean())
    cur_n = float(ctx["cur_hourly"][ctx["cur_hourly"]["timestamp"].dt.hour.isin(hours)]["consumption_l"].mean())
    ratio = cur_n / base_n if base_n > 0 else 0.0

    base_p = ctx["base_hourly"]["pressure_psi"].mean() if "pressure_psi" in ctx["zdf"] else float("nan")
    cur_p = ctx["cur_hourly"]["pressure_psi"].mean() if "pressure_psi" in ctx["zdf"] else float("nan")
    drop = float((base_p - cur_p) / base_p * 100) if base_p == base_p and base_p > 0 and cur_p == cur_p else 0.0

    comp = {"consumption_anomaly": {"HIGH": 40, "MEDIUM": 20}.get(anom["status"], 0),
            "night_flow": 30 if ratio >= 3 else 15 if ratio >= 1.5 else 0,
            "pressure_drop": 20 if drop >= 10 else 10 if drop >= 5 else 0,
            "persistence": 10 if ctx["persist"] >= 3 else 0}
    score = sum(comp.values())
    level = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"

    ev = []
    if anom["status"] != "NORMAL":
        ev.append(f"{anom['deviation_pct']:+.1f}% consumption vs baseline (z={anom['z_score']})")
    if ratio >= 1.5:
        ev.append(f"Night flow (01:00-04:00) {cur_n:,.0f} L/h vs baseline {base_n:,.0f} L/h ({ratio:.1f}x)")
    if drop >= 5:
        ev.append(f"Pressure reduced {drop:.1f}% vs baseline")
    if ctx["persist"] >= 2:
        ev.append(f"Elevated for {ctx['persist']} consecutive days")
    return {"zone": ctx["zone"], "score": score, "level": level, "components": comp,
            "metrics": {"night_flow_ratio": round(ratio, 2), "night_flow_current_lph": round(cur_n, 1),
                        "night_flow_baseline_lph": round(base_n, 1), "pressure_drop_pct": round(drop, 1),
                        "persistent_days": int(ctx["persist"])},
            "evidence": ev, "note": "Prototype risk score; not proof that a physical leak exists."}
