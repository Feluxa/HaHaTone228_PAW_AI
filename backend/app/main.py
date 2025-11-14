from fastapi import FastAPI
from pydantic import BaseModel

from .gigachat_client import ask_gigachat

app = FastAPI(title="Hackathon Reddit AI Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    answer = ask_gigachat(payload.question)
    return {
        "question": payload.question,
        "answer": answer,
    }
