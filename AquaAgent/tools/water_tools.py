"""Water data + consumption + anomaly tools (real calculations, no LLM)."""
import numpy as np
import pandas as pd
from config import settings

_DF = None
_VALIDATION_REPORT = {"total_rows": 0, "valid_rows": 0, "dropped_rows": 0, "reasons": {}}


def validate_and_clean(df):
    """
    Validates incoming telemetry and records detailed drop metrics (FR-03 compliance).
    Returns (cleaned_df, report_dict).
    """
    global _VALIDATION_REPORT
    df = df.copy()
    total = len(df)
    reasons = {"missing_fields": 0, "negative_consumption": 0, "invalid_timestamp": 0, "duplicates": 0}

    df.columns = [c.strip().lower() for c in df.columns]
    
    # Check invalid timestamp
    ts_series = pd.to_datetime(df.get("timestamp"), errors="coerce")
    reasons["invalid_timestamp"] = int(ts_series.isna().sum())
    df["timestamp"] = ts_series

    # Check missing critical columns
    missing_crit = df["timestamp"].isna() | df["zone"].isna() if "zone" in df.columns else df["timestamp"].isna()
    if "consumption_l" in df.columns:
        missing_crit = missing_crit | df["consumption_l"].isna()
    reasons["missing_fields"] = int(missing_crit.sum())
    df = df.dropna(subset=[c for c in ["timestamp", "zone", "consumption_l"] if c in df.columns])

    # Check negative consumption
    if "consumption_l" in df.columns:
        negatives = df["consumption_l"] < 0
        reasons["negative_consumption"] = int(negatives.sum())
        df = df[~negatives]

    # Check duplicates
    if "zone" in df.columns:
        dup_mask = df.duplicated(subset=["timestamp", "zone"])
        reasons["duplicates"] = int(dup_mask.sum())
        df = df.drop_duplicates(subset=["timestamp", "zone"])

    clean_df = df.sort_values("timestamp").reset_index(drop=True)
    report = {
        "total_rows": total,
        "valid_rows": len(clean_df),
        "dropped_rows": total - len(clean_df),
        "reasons": reasons
    }
    _VALIDATION_REPORT = report
    return clean_df, report


def _clean(df):
    clean_df, _ = validate_and_clean(df)
    return clean_df


def get_validation_report():
    """Return the last data validation report."""
    return _VALIDATION_REPORT


def set_dataframe(df):
    """Use an uploaded / test DataFrame instead of the CSV on disk."""
    global _DF
    _DF = _clean(df)


def load_data():
    global _DF
    if _DF is None:
        if not settings.DATA_PATH.exists():
            from data.generate_data import generate
            generate(settings.DATA_PATH)
        _DF = _clean(pd.read_csv(settings.DATA_PATH))
    return _DF


def list_zones():
    return sorted(load_data()["zone"].unique().tolist())


def resolve_zone(name):
    """Case-insensitive zone lookup -> canonical name or None."""
    for z in list_zones():
        if z.lower().replace(" ", "") == str(name).lower().replace(" ", ""):
            return z
    return None


def _split(zone):
    """Shared baseline/current split. Returns (ctx, error_dict)."""
    z = resolve_zone(zone)
    if z is None:
        return None, {"error": f"Unknown zone '{zone}'. Available: {list_zones()}"}
    df = load_data()
    zdf = df[df["zone"] == z].copy()
    zdf["date"] = zdf["timestamp"].dt.normalize()
    g = zdf.groupby("date")
    daily = g.agg(consumption=("consumption_l", "sum"), hours=("consumption_l", "size"))
    daily = daily[daily["hours"] >= 20]            # ignore incomplete days
    n = len(daily)
    if n < 3:
        return None, {"error": f"Insufficient data for {z} ({n} complete days)."}
    recent = settings.RECENT_DAYS if n >= settings.RECENT_DAYS + 5 else 1
    base_idx = daily.index[:-recent]
    base = daily.loc[base_idx, "consumption"]
    cur_date = daily.index[-1]
    cur = float(daily["consumption"].iloc[-1])
    mean, std = float(base.mean()), float(base.std(ddof=1)) or 1.0
    dev = (cur - mean) / mean * 100
    persist = 0
    for v in daily["consumption"].iloc[::-1]:
        if (v - mean) / mean * 100 >= settings.DEV_MED:
            persist += 1
        else:
            break
    ctx = dict(zone=z, zdf=zdf, daily=daily, cur_date=cur_date, cur=cur, base_mean=mean,
               base_std=std, dev=dev, z=(cur - mean) / std, persist=persist,
               base_hourly=zdf[zdf["date"].isin(base_idx)], cur_hourly=zdf[zdf["date"] == cur_date])
    return ctx, None


# ----------------------------- TOOLS ---------------------------------------
def get_water_data(zone=None):
    """Dataset overview (zones, date range, rows) or one zone's latest readings."""
    df = load_data()
    out = {"rows": int(len(df)), "zones": list_zones(),
           "from": str(df["timestamp"].min().date()), "to": str(df["timestamp"].max().date())}
    if zone:
        ctx, err = _split(zone)
        if err:
            return err
        out["zone"] = ctx["zone"]
        out["latest_date"] = str(ctx["cur_date"].date())
        out["latest_daily_consumption_l"] = round(ctx["cur"], 1)
    return out


def calculate_consumption(zone):
    """Total/avg/max/min daily consumption plus baseline vs latest day."""
    ctx, err = _split(zone)
    if err:
        return err
    d = ctx["daily"]["consumption"]
    return {"zone": ctx["zone"], "latest_date": str(ctx["cur_date"].date()),
            "current_l": round(ctx["cur"], 1), "baseline_avg_l": round(ctx["base_mean"], 1),
            "deviation_pct": round(ctx["dev"], 1), "total_l": round(float(d.sum()), 1),
            "avg_daily_l": round(float(d.mean()), 1), "max_daily_l": round(float(d.max()), 1),
            "min_daily_l": round(float(d.min()), 1), "days_analysed": int(len(d))}


def detect_anomaly(zone):
    """Z-score + % deviation vs baseline -> NORMAL / MEDIUM / HIGH."""
    ctx, err = _split(zone)
    if err:
        return err
    dev, z = ctx["dev"], ctx["z"]
    if z >= settings.Z_HIGH and dev >= settings.DEV_HIGH:
        status = "HIGH"
    elif z >= settings.Z_MED and dev >= settings.DEV_MED:
        status = "MEDIUM"
    else:
        status = "NORMAL"
    conf = round(min(0.99, 0.5 + 0.5 * (1 - np.exp(-max(z, 0) / 4))), 2) if status != "NORMAL" else 0.90
    return {"zone": ctx["zone"], "date": str(ctx["cur_date"].date()), "status": status,
            "current_l": round(ctx["cur"], 1), "baseline_avg_l": round(ctx["base_mean"], 1),
            "deviation_pct": round(dev, 1), "z_score": round(float(z), 2),
            "persistent_days": int(ctx["persist"]), "confidence": conf}


def estimate_water_loss(zone):
    """Potential water wastage = actual - expected (SDG 6 efficiency metric)."""
    ctx, err = _split(zone)
    if err:
        return err
    daily = ctx["daily"]["consumption"]
    loss_day = max(0.0, ctx["cur"] - ctx["base_mean"])
    tail = daily.iloc[-ctx["persist"]:] if ctx["persist"] else daily.iloc[0:0]
    total = float(np.maximum(tail - ctx["base_mean"], 0).sum())
    return {"zone": ctx["zone"], "potential_loss_l_per_day": round(loss_day, 1),
            "potential_loss_total_l": round(total, 1), "days_counted": int(ctx["persist"]),
            "excess_pct_of_current": round(loss_day / ctx["cur"] * 100, 1) if ctx["cur"] else 0.0,
            "note": "Estimate vs historical baseline; not a confirmed physical loss."}
