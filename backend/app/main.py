import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from typing import List
from typing import Literal

from .gigachat_client import ask_gigachat
from .vector_store import search_similar


app = FastAPI(title="Hackathon Reddit AI Assistant 288!")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("reddit_rag")


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
    mode: str  # "CODE" или "CHAT"


# Статика и главная страница
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok, 52 Bratuha! Chupapi Munyaya Killer 228! Sber Topchick!"}

def classify_question(query: str) -> Literal["CODE", "CHAT"]:
    """
    Классифицируем вопрос: про код репозитория Reddit или общий разговор.
    Используем GigaChat как классификатор.
    Возвращаем строго 'CODE' или 'CHAT'.
    """
    logger.info("Classifying query: %r", query)

    prompt = f"""
Ты — классификатор запросов для ассистента по коду монорепозитория Reddit.

Твоя задача — отнести вопрос к одному из двух классов:

1) CODE — если вопрос связан с:
   - кодом, функциями, классами, модулями;
   - ошибками, логами, traceback, багами;
   - структурами файлов, конфигами, настройками, базой данных;
   - работой API, контроллеров, роутов, бекенда, авторизации, регистрацией;
   - любыми изменениями/доработками в коде или инфраструктуре проекта.

2) CHAT — если вопрос:
   - не связан с кодом и репозиторием (мемы, жизнь, философия, отношения, шутки);
   - абстрактный разговор, small talk, вопросы не про программирование;
   - общие вопросы про мир, без привязки к репозиторию.

ФОРМАТ ОТВЕТА:
Ответь ОДНИМ СЛОВОМ БЕЗ ОБЪЯСНЕНИЙ:
Либо 'CODE', либо 'CHAT'.

НЕ добавляй никаких комментариев, пояснений или форматирования.

Вопрос пользователя:
\"\"\"{query}\"\"\"
"""

    raw = ask_gigachat(prompt)
    if not raw:
        logger.warning("Empty classification result, fallback to CODE")
        return "CODE"

    normalized = raw.strip().upper()
    logger.info("Raw classification result: %r (normalized: %r)", raw, normalized)

    if "CODE" in normalized and "CHAT" not in normalized:
        return "CODE"
    if "CHAT" in normalized and "CODE" not in normalized:
        return "CHAT"

    # Если модель всё равно что-то намудрила — дефолтимся к CODE,
    # чтобы для техвопросов RAG точно сработал.
    logger.warning("Ambiguous classification %r, fallback to CODE", normalized)
    return "CODE"

@app.post("/ask", response_model=AskResponse)
def ask(payload: Question):
    query = payload.question
    mode = classify_question(query)

    if mode == "CHAT":
        general_prompt = f'''
Ты — дружелюбный ассистент. Ответь кратко и по-человечески.
Вопрос: "{query}"
'''
        answer = ask_gigachat(general_prompt)
        return AskResponse(answer=answer, snippets=[], mode="CHAT")

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
            f"{i}) Файл: {meta['file_path']} (строки {meta['start_line']}-{meta['end_line']})\n```python\n{code}\n```"
        )

    if not snippets:
        fallback_prompt = f'''
Не найдено релевантных фрагментов.
Вопрос: "{query}"
'''
        answer = ask_gigachat(fallback_prompt)
        return AskResponse(answer=answer, snippets=[], mode="CODE")

    context_text = "\n\n".join(context_parts)

    prompt = f'''
Ты — инженерный ассистент по коду большого монорепозитория Reddit.

У ТЕБЯ ЕСТЬ ТОЛЬКО ЭТИ ДАННЫЕ:
- вопрос пользователя (на естественном языке),
- несколько фрагментов кода (functions/classes) с путями к файлам и номерами строк.

ТВОЯ ЗАДАЧА:
1. Понять, что хочет пользователь (например, регистрация, авторизация, «почистить мусор», логирование и т.д.).
2. Найти среди переданных фрагментов те, которые логически связаны с этим запросом.
3. Объяснить, как работает соответствующий код.
4. Подсказать, какие изменения можно внести (где и что примерно менять).

ОЧЕНЬ ВАЖНЫЕ ПРАВИЛА:
- Отвечай ТОЛЬКО на основе предоставленных фрагментов кода.
- НЕЛЬЗЯ придумывать несуществующие файлы, функции или параметры.
- Если информации не хватает — честно напиши, чего именно не хватает, и что ещё стоит поискать в коде.
- Всегда указывай file_path и номера строк для важных мест.
- Если вопрос не про код (или код не найден) — скажи об этом и не фантазируй о структуре репозитория.

ФОРМАТ ОТВЕТА (строго, в Markdown):

## Краткий ответ
1–3 предложения: что происходит и где это реализовано в коде.

## Важные фрагменты кода
Список в виде:
- path/to/file.py:строки A–B — короткое описание фрагмента.

## Детальный разбор
Подробно опиши:
- как работают найденные функции/классы,
- что они принимают,
- как связаны друг с другом,
- какие ветки логики относятся к вопросу.

Используй код-блоки:
```python
# пример функции, важной для анализа
def some_function(...):
    ...
```
Рекомендации по изменению кода

Если пользователь хочет что-то добавить/изменить:

укажи конкретные шаги,

куда вставить новый код,

какую функцию расширить,

какую логику проверить.

Ограничения и что ещё посмотреть

Если данных мало — укажи:

каких файлов или частей кода не хватает,

какие ключевые слова стоит поискать в проекте.

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
\"\"\"{query}\"\"\"
РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОДА (с путями и строками):
{context_text}
Сформулируй ответ строго в указанном выше формате Markdown.
'''
    answer = ask_gigachat(prompt)
    return AskResponse(
    answer=answer,
    snippets=snippets,
    mode="CODE",
)