"""Run this on Day 1 to confirm your IBM credentials work:  python test_watsonx.py"""
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()
model = ModelInference(
    model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"),
    credentials=Credentials(url=os.getenv("WATSONX_URL"), api_key=os.getenv("WATSONX_API_KEY")),
    project_id=os.getenv("WATSONX_PROJECT_ID"))
print(model.generate_text(prompt="Say hello in one short sentence.", params={"max_new_tokens": 30}))
