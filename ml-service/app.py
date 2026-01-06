from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Test Service")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    model: str

@app.post("/ask", response_model=AnswerResponse)
def ask_llm(req: QuestionRequest):
    return {
        "answer": "Sẽ gọi đến Minimax LLM",
        "model": "minimax"
    }
