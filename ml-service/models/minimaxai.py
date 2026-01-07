import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MINIMAX_API_KEY")
BASE_URL = os.getenv("MINIMAX_BASE_URL").rstrip("/")

if not API_KEY or not BASE_URL:
    raise RuntimeError("Missing MINIMAX_API_KEY or MINIMAX_BASE_URL")

CHAT_URL = f"{BASE_URL}/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def generate(prompt: str) -> str:
    payload = {
        "model": "MiniMax-M2.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0
    }

    try:
        resp = requests.post(
            CHAT_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"]

    except requests.HTTPError:
        return f"[MiniMax HTTP Error] {resp.text}"
    except Exception as e:
        return f"[MiniMax Error] {str(e)}"
