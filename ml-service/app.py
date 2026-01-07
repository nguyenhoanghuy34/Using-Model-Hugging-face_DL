from fastapi import FastAPI
from pydantic import BaseModel
from models.minimaxai import generate

app = FastAPI(title="MiniMax API Wrapper")

class AskRequest(BaseModel):
    question: str

@app.post("/askminimax")
def ask_minimax(req: AskRequest):
    answer = generate(req.question)
    return {
        "model": "MiniMax-M2.1",
        "answer": answer
    }
