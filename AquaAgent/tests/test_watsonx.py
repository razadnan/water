"""Test IBM watsonx.ai connectivity. Skips automatically if credentials or package are unavailable."""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

ibm_module = pytest.importorskip("ibm_watsonx_ai", reason="ibm_watsonx_ai package is not installed")

api_key = os.getenv("WATSONX_API_KEY", "")
project_id = os.getenv("WATSONX_PROJECT_ID", "")
url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not (api_key and project_id and url):
    pytest.skip("WATSONX credentials not configured in environment/.env; skipping live API test", allow_module_level=True)


def test_watsonx_live_inference():
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
    model = ModelInference(
        model_id=model_id,
        credentials=Credentials(url=url, api_key=api_key),
        project_id=project_id,
    )
    res = model.generate_text(prompt="Say hello in one short sentence.", params={"max_new_tokens": 20})
    assert isinstance(res, str) and len(res.strip()) > 0
