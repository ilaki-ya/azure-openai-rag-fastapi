from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_core import ask_question

app = FastAPI(title="Azure OpenAI RAG API")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_question(req.question)
    return {"answer": answer}
