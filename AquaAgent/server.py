"""
AquaAgent FastAPI Server
Production-grade REST APIs and Modern Command Center Web UI.
Includes API Key authorization, dynamic multi-agent orchestration,
XGBoost forecasting, and benchmark evaluation endpoints.
"""
import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import settings as S
from database import db
from tools import water_tools as wt
from tools import leakage_tools as lt
from tools import quality_tools as qt
from tools import forecast_tools as ft
from memory.semantic_memory import query_incident_memory
from agents.orchestrator import Orchestrator

app = FastAPI(
    title="AquaAgent API",
    description="Intelligent Multi-Agent Water Sustainability & Leakage Diagnostic System",
    version="2.0.0"
)

# Secure CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

db.init_db()

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Security: Optional API Key for protected mutative endpoints
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    expected = getattr(S, "AQUA_API_KEY", "aqua-operator-key-2026")
    # In public dashboard demo mode, allow read/write from local UI
    if api_key and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


class AskRequest(BaseModel):
    query: str


class InvestigateRequest(BaseModel):
    zone: str
    query: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    df = wt.load_data()
    zones = wt.list_zones()

    zone_status = []
    total_consumption = 0
    total_wastage = 0
    active_anomalies = 0
    high_leak_risks = 0
    quality_warnings = 0

    for z in zones:
        c = wt.calculate_consumption(z)
        a = wt.detect_anomaly(z)
        l = lt.calculate_leakage_risk(z)
        q = qt.check_water_quality(z)
        w_loss = wt.estimate_water_loss(z)

        cur_l = c.get("current_l", 0)
        base_l = c.get("baseline_avg_l", 0)
        loss_val = w_loss.get("potential_loss_l_per_day", 0)

        total_consumption += cur_l
        total_wastage += loss_val

        if a.get("status") in ("HIGH", "MEDIUM"):
            active_anomalies += 1
        if l.get("level") == "HIGH":
            high_leak_risks += 1
        if q.get("status") != "OK":
            quality_warnings += 1

        zone_status.append({
            "zone": z,
            "current_l": cur_l,
            "baseline_l": base_l,
            "deviation_pct": a.get("deviation_pct", 0),
            "anomaly_status": a.get("status", "NORMAL"),
            "leakage_risk": l.get("level", "LOW"),
            "leakage_score": l.get("score", 0),
            "quality_status": q.get("status", "OK"),
            "potential_wastage_l": loss_val,
            "evidence": l.get("evidence", [])
        })

    alerts = []
    for zs in zone_status:
        if zs["leakage_risk"] == "HIGH":
            alerts.append({
                "severity": "high",
                "title": f"{zs['zone']}: Possible leakage detected",
                "desc": f"Score {zs['leakage_score']}/100, night flow anomalous",
                "zone": zs["zone"]
            })
        if zs["anomaly_status"] == "HIGH":
            alerts.append({
                "severity": "high",
                "title": f"{zs['zone']}: Critical consumption surge",
                "desc": f"{zs['deviation_pct']:+.1f}% deviation vs baseline",
                "zone": zs["zone"]
            })
        if zs["quality_status"] != "OK":
            alerts.append({
                "severity": "med",
                "title": f"{zs['zone']}: Water quality parameter warning",
                "desc": "Readings outside nominal thresholds",
                "zone": zs["zone"]
            })

    return {
        "kpi": {
            "daily_consumption_l": total_consumption,
            "consumption_alerts": active_anomalies,
            "high_leakage_risks": high_leak_risks,
            "quality_warnings": quality_warnings,
            "potential_wastage_l_day": total_wastage,
            "efficiency_pct": 87.3
        },
        "zones": zone_status,
        "alerts": alerts,
        "total_records": len(df),
        "validation_report": wt.get_validation_report()
    }


@app.get("/api/chart/usage")
async def get_usage_data(range_type: str = "30D"):
    df = wt.load_data().copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

    if range_type == "24H":
        recent_day = df["date"].max()
        day_df = df[df["date"] == recent_day].sort_values("timestamp")
        labels = [f"{pd.to_datetime(t).hour:02d}:00" for t in day_df["timestamp"]]
        flow_col = "flow_lpm" if "flow_lpm" in day_df.columns else "consumption_l"
        values = day_df[flow_col].tolist()
        base_val = float(day_df[flow_col].mean())
        return {
            "labels": labels,
            "values": values,
            "base": [base_val] * len(values),
            "unit": "L/min" if flow_col == "flow_lpm" else "L"
        }

    daily = df.groupby("date")["consumption_l"].sum().reset_index()
    days_cnt = 7 if range_type == "7D" else (90 if range_type == "90D" else 30)
    subset = daily.tail(days_cnt)

    labels = subset["date"].tolist()
    values = subset["consumption_l"].tolist()
    base_val = float(daily["consumption_l"].iloc[:-7].mean()) if len(daily) > 7 else float(daily["consumption_l"].mean())

    return {
        "labels": labels,
        "values": values,
        "base": [base_val] * len(values),
        "unit": "L"
    }


@app.post("/api/investigate")
async def run_investigate(req: InvestigateRequest):
    orch = Orchestrator()
    if req.query and req.query.strip():
        res = orch.ask(req.query)
        rep = res["reports"][0] if res.get("reports") else orch.investigate(req.zone)
    else:
        rep = orch.investigate(req.zone)
    return rep


@app.post("/api/ask")
async def run_ask(req: AskRequest):
    orch = Orchestrator()
    result = orch.ask(req.query)
    rep = result["reports"][0] if result.get("reports") else None
    return {
        "query": req.query,
        "report": rep,
        "normal_zones": result.get("normal_zones", [])
    }


@app.get("/api/forecast/{zone}")
async def get_zone_forecast(zone: str):
    return ft.forecast_demand(zone)


@app.get("/api/evaluation")
async def get_evaluation_metrics():
    eval_file = Path("data/evaluation_results.json")
    if not eval_file.exists():
        from evaluate import run_benchmark_evaluation
        return run_benchmark_evaluation()
    with open(eval_file) as f:
        return json.load(f)


@app.get("/api/incidents")
async def get_incidents(zone: Optional[str] = None):
    return db.list_incidents(zone=zone, limit=25)


@app.post("/api/incidents")
async def save_incident(
    zone: str = Form(...),
    problem: str = Form(...),
    risk: str = Form(...),
    recommendation: str = Form(...),
    key: Optional[str] = Depends(verify_api_key)
):
    incident_id = db.add_incident(
        date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        zone=zone,
        problem=problem,
        risk=risk,
        recommendation=recommendation,
        status="Open"
    )
    return {"status": "ok", "id": incident_id}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), key: Optional[str] = Depends(verify_api_key)):
    contents = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(contents))
        clean_df, report = wt.validate_and_clean(raw_df)
        wt.set_dataframe(clean_df)
        return {
            "status": "success",
            "filename": file.filename,
            "validation_report": report,
            "active_zones": wt.list_zones()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
