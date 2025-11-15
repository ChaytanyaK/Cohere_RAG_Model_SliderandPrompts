import os
import json
import uuid
from datetime import datetime

from config import CHAT_DATA_DIR

def save_chat_history(history,  path):
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure path exists
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_chat_history(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
