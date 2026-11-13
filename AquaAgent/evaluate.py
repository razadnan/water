"""
Benchmark Evaluation Framework for Leakage & Anomaly Detection.
Runs multiple detection algorithms against ground-truth benchmark labels:
1. AquaAgent Heuristic Scorer (Consumption + Night Flow + Pressure + Persistence)
2. Scikit-Learn IsolationForest Anomaly Detector (Honest ML baseline: trained strictly on pre-leak history)
3. Rolling-Median Statistical Threshold Baseline

Also computes:
- Signal Ablation Study: examines individual and partial contributions (Night-Flow alone, Night-Flow + Pressure)
- False Alarm Rate on normal control weeks
- Precision, Recall, F1
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import settings


def run_benchmark_evaluation(data_csv="data/water_data.csv", labels_json="data/benchmark_labels.json"):
    data_path = Path(data_csv)
    labels_path = Path(labels_json)

    if not data_path.exists() or not labels_path.exists():
        from data.benchmark_generator import generate_benchmark_data
        generate_benchmark_data()

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    with open(labels_path) as f:
        ground_truth = json.load(f)

    events = ground_truth["events"]
    df["date"] = df["timestamp"].dt.date
    zones = df["zone"].unique()

    # Ground truth mapping: (zone, date_str) -> is_leak (bool)
    gt_map = {}
    for z in zones:
        for d in df[df["zone"] == z]["date"].unique():
            gt_map[(z, str(d))] = False

    for ev in events:
        if ev.get("is_leak"):
            st_d = datetime.strptime(ev["start"], "%Y-%m-%d %H:%M:%S").date()
            en_d = datetime.strptime(ev["end"], "%Y-%m-%d %H:%M:%S").date()
            cur = st_d
            while cur <= en_d:
                gt_map[(ev["zone"], str(cur))] = True
                cur += timedelta(days=1)

    y_true = []
    y_pred_aqua = []
    y_pred_night_only = []
    y_pred_night_pressure = []

    from tools import water_tools as wt, leakage_tools as lt

    for z in zones:
        zdf = df[df["zone"] == z].copy()
        dates = sorted(zdf["date"].unique())
        for d in dates:
            is_leak_gt = gt_map.get((z, str(d)), False)
            y_true.append(1 if is_leak_gt else 0)

            # Sequential realistic evaluation up to date d
            sub = zdf[zdf["date"] <= d]
            if len(sub["date"].unique()) >= 5:
                wt.set_dataframe(sub)
                leak_res = lt.calculate_leakage_risk(z)
                pred_leak = 1 if leak_res.get("level") in ("HIGH", "MEDIUM") else 0
                
                # Ablation 1: Night flow only (ratio >= 2.0)
                n_ratio = leak_res.get("metrics", {}).get("night_flow_ratio", 1.0)
                pred_night = 1 if n_ratio >= 2.0 else 0
                
                # Ablation 2: Night flow + pressure drop (ratio >= 1.5 AND drop >= 3.0)
                p_drop = leak_res.get("metrics", {}).get("pressure_drop_pct", 0.0)
                pred_np = 1 if (n_ratio >= 1.5 and p_drop >= 3.0) else 0
            else:
                pred_leak = 0
                pred_night = 0
                pred_np = 0

            y_pred_aqua.append(pred_leak)
            y_pred_night_only.append(pred_night)
            y_pred_night_pressure.append(pred_np)

    # -------------------------------------------------------------
    # Model 2: IsolationForest (Strict Unsupervised ML Baseline)
    # -------------------------------------------------------------
    # Note: Trained ONLY on first 15 days of each zone (clean baseline without leak knowledge)
    # Contamination set to 0.05 default (not tuned to ground truth)
    y_pred_iforest = []
    for z in zones:
        zdf = df[df["zone"] == z].copy()
        features = zdf[["consumption_l", "flow_lpm", "pressure_psi"]].fillna(0)
        train_len = min(len(features), 24 * 15)  # strictly clean first 15 days
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(features.iloc[:train_len])
        preds = clf.predict(features)  # -1 anomaly, 1 normal
        zdf["is_anom"] = (preds == -1).astype(int)

        daily_anoms = zdf.groupby("date")["is_anom"].mean()
        for d in sorted(zdf["date"].unique()):
            is_anom_day = 1 if daily_anoms.get(d, 0) >= 0.25 else 0
            y_pred_iforest.append(is_anom_day)

    # -------------------------------------------------------------
    # Model 3: Rolling Median Baseline
    # -------------------------------------------------------------
    y_pred_median = []
    for z in zones:
        zdf = df[df["zone"] == z].groupby("date")["consumption_l"].sum().reset_index()
        zdf["rolling_median"] = zdf["consumption_l"].rolling(7, min_periods=3).median()
        zdf["dev"] = (zdf["consumption_l"] - zdf["rolling_median"]) / zdf["rolling_median"] * 100
        for _, row in zdf.iterrows():
            pred = 1 if row["dev"] > 20.0 else 0
            y_pred_median.append(pred)

    def calc_metrics(y_t, y_p):
        yt = np.array(y_t)
        yp = np.array(y_p)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "false_alarm_rate_pct": round(fpr * 100.0, 2)
        }

    m_aqua = calc_metrics(y_true, y_pred_aqua)
    m_iforest = calc_metrics(y_true, y_pred_iforest)
    m_median = calc_metrics(y_true, y_pred_median)
    m_night_only = calc_metrics(y_true, y_pred_night_only)
    m_night_pressure = calc_metrics(y_true, y_pred_night_pressure)

    eval_summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_days_evaluated": len(y_true),
        "total_leak_days": sum(y_true),
        "models": {
            "AquaAgent_MultiSignal": m_aqua,
            "ScikitLearn_IsolationForest": m_iforest,
            "Rolling_Median_Threshold": m_median
        },
        "ablation_study": {
            "Night_Flow_Only": m_night_only,
            "Night_Flow_Plus_Pressure": m_night_pressure,
            "Full_MultiSignal_Fusion": m_aqua
        }
    }

    eval_out = Path("data/evaluation_results.json")
    with open(eval_out, "w") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


if __name__ == "__main__":
    res = run_benchmark_evaluation()
    print("\n================== BENCHMARK EVALUATION RESULTS ==================")
    for model_name, metrics in res["models"].items():
        print(f"\nModel: {model_name}")
        print(f"  Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate_pct']:.2f}% (FP: {metrics['false_positives']}, FN: {metrics['false_negatives']})")
    
    print("\n----------------- SIGNAL ABLATION STUDY -----------------")
    for signal_name, metrics in res["ablation_study"].items():
        print(f"  {signal_name:25s} -> Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}")
    print("==================================================================")
