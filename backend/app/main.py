from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from .gigachat_client import ask_gigachat
from .vector_store import search_similar


app = FastAPI(title="Hackathon Reddit AI Assistant")


class Question(BaseModel):
    question: str


class Snippet(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    code: str


class AskResponse(BaseModel):
    answer: str
    snippets: List[Snippet]


# Статика и главная страница
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: Question):
    query = payload.question

    # 1. Ищем похожие чанки кода в векторной базе
    matches = search_similar(query, k=5)

    snippets: List[Snippet] = []
    context_parts: List[str] = []

    for i, m in enumerate(matches, start=1):
        meta = m["metadata"]
        code = m["document"]

        snippet = Snippet(
            file_path=meta["file_path"],
            start_line=meta["start_line"],
            end_line=meta["end_line"],
            code=code,
        )
        snippets.append(snippet)

        context_parts.append(
            f"{i}) Файл: {meta['file_path']} (строки {meta['start_line']}-{meta['end_line']})\n"
            f"```python\n{code}\n```"
        )

    if not snippets:
        answer = ask_gigachat(
            f"Вопрос по коду репозитория Reddit (но в индексе ничего не нашли): {query}"
        )
        return AskResponse(answer=answer, snippets=[])

    context_text = "\n\n".join(context_parts)

    prompt = f"""
Ты — помощник по коду большого монорепозитория Reddit. 
Твоя задача — помочь разработчику понять, какие части кода относятся к его вопросу 
и что с ними можно сделать.

ВНИМАНИЕ:
- Отвечай, опираясь ТОЛЬКО на предоставленные фрагменты кода.
- Обязательно указывай, какие файлы и функции задействованы (file_path, имя функции, строки).
- Если спрашивают что-то вроде "почистить мусор", "удалить старые данные" и т.п.,
  ищи функции, которые логически связаны (clear_*, delete_*, cleanup_* и т.п.) 
  и объясняй, как их вызывать/менять.

Вопрос пользователя:
\"\"\"{query}\"\"\"

Ниже — релевантные фрагменты кода:

{context_text}

Сформулируй понятный и структурированный ответ:
1) Опиши, какие функции/классы связаны с вопросом.
2) Объясни, как они работают.
3) Подскажи, какие изменения можно внести в код, если нужно что-то изменить.
"""

    answer = ask_gigachat(prompt)

    return AskResponse(
        answer=answer,
        snippets=snippets,
    )
