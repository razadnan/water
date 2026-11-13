"""Short-term (per investigation) and long-term (incident history) memory."""
from database import db


class ShortTermMemory:
    """Context shared between agents during ONE investigation."""

    def __init__(self):
        self._store = {}

    def set(self, key, value):
        self._store[key] = value

    def get(self, key, default=None):
        return self._store.get(key, default)

    def snapshot(self):
        return dict(self._store)


class LongTermMemory:
    """Persistent incident history backed by SQLite."""

    @staticmethod
    def recall(zone, limit=5):
        return db.list_incidents(zone, limit)

    @staticmethod
    def remember(date, zone, problem, risk, recommendation, resolution="", status="Open"):
        return db.add_incident(date, zone, problem, risk, recommendation, resolution, status)
