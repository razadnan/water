"""Tools that read/write long-term incident memory."""
from memory.memory import LongTermMemory
from database import db


def get_historical_incidents(zone, limit=5):
    return {"zone": zone, "incidents": LongTermMemory.recall(zone, limit)}


def save_incident(zone, problem, risk, recommendation, date, resolution="", status="Open"):
    if db.find_open_incident(zone, date):
        return {"saved": False, "reason": "An open incident for this zone/date already exists."}
    new_id = LongTermMemory.remember(date, zone, problem, risk, recommendation, resolution, status)
    return {"saved": True, "incident_id": new_id}
