"""Water-quality threshold checks (limits configurable in config/settings.py)."""
from config import settings
from tools.water_tools import _split


def check_water_quality(zone):
    ctx, err = _split(zone)
    if err:
        return err
    cur = ctx["cur_hourly"]
    alerts, readings = [], {}
    for col, (lo, hi) in settings.QUALITY_LIMITS.items():
        if col not in cur or cur[col].dropna().empty:
            continue
        mean = float(cur[col].mean())
        out_hours = int(((cur[col] < lo) | (cur[col] > hi)).sum())
        readings[col] = round(mean, 2)
        if mean < lo or mean > hi or out_hours >= 6:
            severe = col == "ph" and (mean < lo - 0.5 or mean > hi + 0.5)
            alerts.append({"parameter": col, "mean": round(mean, 2), "allowed": [lo, hi],
                           "hours_out_of_range": out_hours,
                           "severity": "CRITICAL" if severe else "WARNING"})
    status = "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts) else "WARNING" if alerts else "OK"
    return {"zone": ctx["zone"], "date": str(ctx["cur_date"].date()), "status": status,
            "readings": readings, "alerts": alerts}
