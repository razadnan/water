"""Tool registry: name -> callable + description the LLM sees."""
import inspect
from tools import water_tools as w, leakage_tools as l, quality_tools as q, memory_tools as m, report_tools as r, forecast_tools as f

TOOLS = {
    "get_water_data": (w.get_water_data, "Dataset overview, or latest reading for a zone. args: zone (optional)"),
    "calculate_consumption": (w.calculate_consumption, "Total/avg/max/min and baseline vs latest-day consumption. args: zone"),
    "detect_anomaly": (w.detect_anomaly, "Z-score and % deviation anomaly check (NORMAL/MEDIUM/HIGH). args: zone"),
    "calculate_leakage_risk": (l.calculate_leakage_risk, "Leakage risk score 0-100 from consumption, night flow, pressure, persistence. args: zone"),
    "estimate_water_loss": (w.estimate_water_loss, "Potential water wastage in litres vs baseline. args: zone"),
    "check_water_quality": (q.check_water_quality, "pH/turbidity threshold check. args: zone"),
    "forecast_demand": (f.forecast_demand, "XGBoost ML demand forecast and evaluation metrics. args: zone"),
    "get_historical_incidents": (m.get_historical_incidents, "Past incidents for a zone from long-term memory. args: zone"),
    "save_incident": (m.save_incident, "Store an incident in long-term memory."),
    "generate_report": (r.generate_report, "Create a markdown report from an investigation."),
}


def describe(names):
    return "\n".join(f"- {n}: {TOOLS[n][1]}" for n in names)


def call_tool(name, **kwargs):
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'"}
    fn = TOOLS[name][0]
    params = inspect.signature(fn).parameters
    clean = {k: v for k, v in kwargs.items() if k in params}
    try:
        return fn(**clean)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # tools must never crash an agent
        return {"error": f"{name} failed: {e}"}
