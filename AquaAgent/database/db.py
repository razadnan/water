"""SQLite store for incident history (long-term memory backend)."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from config import settings

_path = Path(settings.DB_PATH)
_ready = set()

SEED = [("2026-08-12", "Zone B", "Pipeline leakage", "HIGH",
         "Inspect pipeline and valves", "Valve replacement", "Resolved")]


def set_path(p):
    global _path
    _path = Path(p)


@contextmanager
def _conn():
    _path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(seed=True):
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS incidents(
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, zone TEXT, problem TEXT,
            risk TEXT, recommendation TEXT, resolution TEXT, status TEXT)""")
        if seed and c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 0:
            c.executemany("INSERT INTO incidents(date,zone,problem,risk,recommendation,resolution,status)"
                          " VALUES (?,?,?,?,?,?,?)", SEED)
    _ready.add(str(_path))


def _ensure():
    if str(_path) not in _ready:
        init_db()


def add_incident(date, zone, problem, risk, recommendation, resolution="", status="Open"):
    _ensure()
    with _conn() as c:
        cur = c.execute("INSERT INTO incidents(date,zone,problem,risk,recommendation,resolution,status)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (date, zone, problem, risk, recommendation, resolution, status))
        return cur.lastrowid


def list_incidents(zone=None, limit=10):
    _ensure()
    with _conn() as c:
        if zone:
            rows = c.execute("SELECT * FROM incidents WHERE zone=? ORDER BY date DESC, id DESC LIMIT ?",
                             (zone, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM incidents ORDER BY date DESC, id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def find_open_incident(zone, date):
    _ensure()
    with _conn() as c:
        r = c.execute("SELECT * FROM incidents WHERE zone=? AND date=? AND status='Open'", (zone, date)).fetchone()
    return dict(r) if r else None
