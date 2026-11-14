import os
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("FINANCE_DATA_DIR", "data/kaggle")
ARTIFACT_DIR = os.getenv("STAIR_ARTIFACT_DIR", "artifacts")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "")
