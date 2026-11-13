"""
Simulate realistic water distribution telemetry with ground-truth labels.
Labels are stored in a separate file (benchmark_labels.json) that detection
algorithms never see at test time.

Supports injection of:
- Incipient and burst leaks (varying sizes, times, durations)
- Legitimate non-leak anomalies (holidays/high demand, meter drift, sensor noise, operational pump changes)
- Normal clean weeks for false-positive validation
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PROFILE = np.array([0.3, 0.25, 0.25, 0.3, 0.6, 1.5, 3.5, 5.5, 5.5, 4.5, 4, 3.8,
                        4, 3.8, 3.5, 3.6, 4, 5, 6, 6.5, 5.5, 4, 2.2, 1.0])
HOURLY_WEIGHTS = RAW_PROFILE / RAW_PROFILE.sum()

ZONES_META = {
    "Zone A": {"base_m3_day": 60.0, "nom_pressure_psi": 55.0, "ph": 7.3, "turbidity": 1.8},
    "Zone B": {"base_m3_day": 85.0, "nom_pressure_psi": 52.0, "ph": 7.2, "turbidity": 2.0},
    "Zone C": {"base_m3_day": 70.0, "nom_pressure_psi": 48.0, "ph": 7.1, "turbidity": 2.2},
    "Zone D": {"base_m3_day": 45.0, "nom_pressure_psi": 50.0, "ph": 7.4, "turbidity": 1.5},
}


def generate_benchmark_data(output_dir="data", days=60, seed=101, end_date="2026-11-12"):
    """
    Generates a 60-day dataset across 4 zones with distinct, realistic test cases:
    - Zone A: Normal baseline throughout (for false alarm evaluation)
    - Zone B: Burst pipe leak starting on day 52 (severe night flow, pressure drop)
    - Zone C: Small incipient leak starting day 45 + water quality drift
    - Zone D: Legitimate non-leak operational surge (festival/event on days 30-31, normal night flow)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days - 1)
    date_range = [start_dt + timedelta(days=i) for i in range(days)]

    telemetry_rows = []
    ground_truth_events = []

    # Inject benchmark scenarios
    # 1. Zone B Leak (Burst pipe starting at day 52)
    b_leak_start = start_dt + timedelta(days=52, hours=2)
    b_leak_end = end_dt + timedelta(days=1)
    ground_truth_events.append({
        "zone": "Zone B",
        "type": "pipe_burst_leak",
        "start": b_leak_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": b_leak_end.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": "HIGH",
        "magnitude_pct": 32.0,
        "is_leak": True
    })

    # 2. Zone C Incipient Leak (Small crack starting day 45)
    c_leak_start = start_dt + timedelta(days=45, hours=1)
    c_leak_end = end_dt + timedelta(days=1)
    ground_truth_events.append({
        "zone": "Zone C",
        "type": "incipient_leak",
        "start": c_leak_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": c_leak_end.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": "MEDIUM",
        "magnitude_pct": 14.0,
        "is_leak": True
    })

    # 3. Zone D Non-Leak Demand Surge (Community event, normal night flow, no pressure drop)
    d_event_start = start_dt + timedelta(days=30, hours=8)
    d_event_end = start_dt + timedelta(days=32, hours=20)
    ground_truth_events.append({
        "zone": "Zone D",
        "type": "legitimate_demand_surge",
        "start": d_event_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": d_event_end.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": "NON_LEAK",
        "magnitude_pct": 28.0,
        "is_leak": False
    })

    for zone, meta in ZONES_META.items():
        base_litres_day = meta["base_m3_day"] * 1000.0

        for di, dt in enumerate(date_range):
            # Day of week variation (weekends slightly different)
            dow_mult = 1.05 if dt.weekday() >= 5 else 0.98
            weather_temp = 22.0 + 5.0 * np.sin(di / 10.0) + rng.normal(0, 0.8)

            for h in range(24):
                curr_time = dt + timedelta(hours=h)
                hour_fraction = HOURLY_WEIGHTS[h]

                # Base expected consumption
                cons_l = base_litres_day * dow_mult * hour_fraction * rng.normal(1.0, 0.04)
                pressure = meta["nom_pressure_psi"] + rng.normal(0, 0.4)
                ph = meta["ph"] + rng.normal(0, 0.05)
                turbidity = meta["turbidity"] + rng.normal(0, 0.1)

                is_leak_hour = False
                event_type = "NORMAL"

                # Check Zone B Leak injection
                if zone == "Zone B" and curr_time >= b_leak_start:
                    leak_lph = (base_litres_day * 0.32 / 24.0) * rng.normal(1.0, 0.02)
                    cons_l += leak_lph
                    pressure -= 7.5 + rng.normal(0, 0.3)  # Significant pressure drop
                    is_leak_hour = True
                    event_type = "LEAK"

                # Check Zone C Incipient Leak injection
                if zone == "Zone C" and curr_time >= c_leak_start:
                    leak_lph = (base_litres_day * 0.14 / 24.0) * rng.normal(1.0, 0.03)
                    cons_l += leak_lph
                    pressure -= 1.8 + rng.normal(0, 0.2)
                    is_leak_hour = True
                    event_type = "LEAK"
                    # Quality anomaly during incipient leak
                    ph = 5.85 + rng.normal(0, 0.05)
                    turbidity = 3.8 + rng.normal(0, 0.2)

                # Check Zone D legitimate surge (Daytime only surge, night flow stays normal)
                if zone == "Zone D" and d_event_start <= curr_time <= d_event_end:
                    if 8 <= h <= 21:
                        cons_l *= 1.35
                    event_type = "PLANNED_SURGE"

                flow_lpm = max(cons_l / 60.0 + rng.normal(0, 0.5), 0.0)

                telemetry_rows.append({
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "zone": zone,
                    "consumption_l": round(float(cons_l), 1),
                    "flow_lpm": round(float(flow_lpm), 2),
                    "pressure_psi": round(float(pressure), 2),
                    "ph": round(float(ph), 2),
                    "turbidity_ntu": round(float(max(turbidity, 0.1)), 2),
                    "temperature_c": round(float(weather_temp), 1)
                })

    df = pd.DataFrame(telemetry_rows)
    csv_file = out_path / "water_data.csv"
    df.to_csv(csv_file, index=False)

    labels_file = out_path / "benchmark_labels.json"
    with open(labels_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_records": len(df),
            "zones": list(ZONES_META.keys()),
            "events": ground_truth_events
        }, f, indent=2)

    return csv_file, labels_file


if __name__ == "__main__":
    c, l = generate_benchmark_data()
    print(f"Generated {c} and ground truth {l}")
