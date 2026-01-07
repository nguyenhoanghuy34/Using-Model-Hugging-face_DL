import os
from minimaxai import MinimaxClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

MINIMAX_KEY = os.environ.get("MINIMAX_KEY")
client = MinimaxClient(api_key=MINIMAX_KEY)

def generate(prompt: str) -> str:
    try:
        response = client.generate(prompt=prompt, max_tokens=200)
        return response.output_text
    except Exception as e:
        return f"[Lỗi gọi Minimax]: {e}"
