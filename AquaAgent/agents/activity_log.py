"""Shared agent activity log (powers the 'AI Investigation' dashboard panel)."""
from datetime import datetime


class ActivityLog:
    def __init__(self):
        self.events = []

    def add(self, agent, event, detail=""):
        self.events.append({"time": datetime.now().strftime("%H:%M:%S"),
                            "agent": agent, "event": event, "detail": str(detail)})

    def rows(self):
        return list(self.events)
