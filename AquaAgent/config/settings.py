"""Central configuration. Secrets come from environment variables / .env only."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("AQUA_DATA_PATH", BASE_DIR / "data" / "water_data.csv"))
DB_PATH = Path(os.getenv("AQUA_DB_PATH", BASE_DIR / "database" / "aquaagent.db"))

# ---- IBM watsonx.ai (leave blank to run in offline/rule-based mode) ----
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
FORCE_OFFLINE = os.getenv("AQUA_FORCE_OFFLINE", "0") == "1"

# ---- Analysis parameters ----
RECENT_DAYS = 7              # newest N days are excluded from the baseline
NIGHT_HOURS = (1, 2, 3)      # 01:00-04:00 night-flow window
DEV_HIGH, DEV_MED = 25.0, 15.0   # % deviation thresholds
Z_HIGH, Z_MED = 3.0, 2.0         # z-score thresholds

# Water-quality limits (configurable per use case; defaults are illustrative)
QUALITY_LIMITS = {"ph": (6.5, 8.5), "turbidity_ntu": (0.0, 4.0)}
