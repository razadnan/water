import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def fresh_env(tmp_path):
    from database import db
    from tools import water_tools
    from data.generate_data import generate
    db.set_path(tmp_path / "test.db")
    water_tools.set_dataframe(generate(tmp_path / "data.csv"))
    yield
