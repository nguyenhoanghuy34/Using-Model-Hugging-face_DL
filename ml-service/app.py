from fastapi import FastAPI
from pydantic import BaseModel
from models.minimaxai import generate  # wrapper Minimax

app = FastAPI(title="LLM Test Service")

# ===== Schema =====
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    model: str

# ===== API =====
@app.post("/askminimax", response_model=AnswerResponse)
def ask_minimax(req: QuestionRequest):
    """
    API test Minimax LLM.
    """
    output = generate(req.question)
    return {
        "answer": output,
        "model": "minimax"
    }
