from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.embeddings.gigachat import GigaChatEmbeddings


# Загружаем .env из backend
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

_embeddings_client: GigaChatEmbeddings | None = None


def get_embeddings_client() -> GigaChatEmbeddings:
    """
    Ленивая инициализация клиента GigaChat Embeddings.
    """
    global _embeddings_client

    if _embeddings_client is not None:
        return _embeddings_client

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "False").lower() == "true"

    if not credentials:
        raise RuntimeError("Не задан GIGACHAT_CREDENTIALS в .env")

    _embeddings_client = GigaChatEmbeddings(
        credentials=credentials,
        scope=scope,
        verify_ssl_certs=verify_ssl,
    )
    return _embeddings_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Эмбеддинги для списка текстов (списка чанков кода).
    """
    client = get_embeddings_client()
    return client.embed_documents(texts)


def embed_text(text: str) -> List[float]:
    """
    Эмбеддинг для одного текста (например, вопрос пользователя).
    """
    client = get_embeddings_client()
    return client.embed_query(text)


if __name__ == "__main__":
    # Быстрый тест
    v = embed_text("test embedding from GigaChat")
    print("Длина эмбеддинга:", len(v))
    print("Первые 5 значений:", v[:5])