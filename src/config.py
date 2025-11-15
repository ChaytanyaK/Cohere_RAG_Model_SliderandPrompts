import os
from pathlib import Path

import cohere
from dotenv import load_dotenv
from openai import OpenAI

# ==== PATHS ====
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHAT_DIR = DATA_DIR / "chat"
ASSETS_DIR = BASE_DIR / "assets"

PDF_FOLDER = RAW_DIR / "source_docs"
IMG_FOLDER = PROCESSED_DIR / "images"
VECTOR_STORE_DIR = PROCESSED_DIR / "vector_store"
HASHES_FOLDER = CACHE_DIR / "hashes"
CHAT_DATA_DIR = CHAT_DIR / "sessions"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "image_index.faiss"
FILENAME_MAP_PATH = VECTOR_STORE_DIR / "image_filenames.pkl"
PDF_HASH_FILE = "pdf_hashes.json"
MODEL_NAME = "embed-v4.0"

BACKGROUND_IMAGE_PATH = ASSETS_DIR / "backgrounds" / "world_bank.jpg"
USER_AVATAR_PATH = ASSETS_DIR / "ui" / "boy.png"
BOT_AVATAR_PATH = ASSETS_DIR / "ui" / "assistant.png"

for path in [
    PDF_FOLDER,
    IMG_FOLDER,
    VECTOR_STORE_DIR,
    HASHES_FOLDER,
    CHAT_DATA_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Load API keys
load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
