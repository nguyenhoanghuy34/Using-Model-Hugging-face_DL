from fastapi import FastAPI
from pydantic import BaseModel
from models.minimaxai import generate       
from models.youtullm import generate_youtu     

app = FastAPI(title="MiniMax + Youtu API Wrapper")

class AskRequest(BaseModel):
    question: str

# API cũ
@app.post("/askminimax")
def ask_minimax(req: AskRequest):
    answer = generate(req.question)
    return {
        "model": "MiniMax-M2.1",
        "answer": answer
    }

# API mới cho Youtu-LLM
@app.post("/askyoutu")
def ask_youtu(req: AskRequest):
    answer = generate_youtu(req.question, max_new_tokens=10)  # giới hạn 10 token
    return {
        "model": "Youtu-LLM-2B",
        "answer": answer
    }

