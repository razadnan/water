"""Synthetic hourly sensor data for 3 zones (30 days). Zone B has an injected leak."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = np.array([0.3, 0.25, 0.25, 0.3, 0.6, 1.5, 3.5, 5.5, 5.5, 4.5, 4, 3.8,
                4, 3.8, 3.5, 3.6, 4, 5, 6, 6.5, 5.5, 4, 2.2, 1.0])
PROFILE = RAW / RAW.sum()

ZONES = {
    "Zone A": dict(base=60000, pressure=55, ph=7.3, leak=0.0, leak_days=0, pressure_drop=0.0),
    "Zone B": dict(base=82000, pressure=52, ph=7.2, leak=0.34, leak_days=4, pressure_drop=0.15),
    "Zone C": dict(base=70000, pressure=48, ph=7.1, leak=0.15, leak_days=2, pressure_drop=0.03,
                   ph_event_days=2),
}


def generate(path, days=30, seed=42, end="2026-11-12"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    rows = []
    for zone, cfg in ZONES.items():
        for di, date in enumerate(dates):
            from_end = days - 1 - di
            daily_factor = rng.normal(1, 0.025)
            leaking = from_end < cfg["leak_days"]
            ph_event = from_end < cfg.get("ph_event_days", 0)
            for h in range(24):
                base_h = cfg["base"] * daily_factor * PROFILE[h] * rng.normal(1, 0.05)
                extra = cfg["base"] * cfg["leak"] / 24 * rng.normal(1, 0.03) if leaking else 0.0
                cons = max(base_h + extra, 0.0)
                pressure = cfg["pressure"] * (1 - (cfg["pressure_drop"] if leaking else 0)) + rng.normal(0, 0.6)
                ph = (5.9 if ph_event else cfg["ph"]) + rng.normal(0, 0.08)
                from datetime import timedelta
                dt = date.to_pydatetime() + timedelta(hours=int(h))
                rows.append(dict(
                    timestamp=dt, zone=zone,
                    consumption_l=round(cons, 1), flow_lpm=round(cons / 60 + rng.normal(0, 2), 1),
                    pressure_psi=round(pressure, 2), ph=round(ph, 2),
                    turbidity_ntu=round(abs(rng.normal(2.0, 0.2)), 2)))
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "water_data.csv"
    print(f"Wrote {len(generate(out))} rows -> {out}")
