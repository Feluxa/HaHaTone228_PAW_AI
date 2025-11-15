from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

from .indexing import CodeChunk
from .embeddings import embed_texts, embed_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"


MAX_CHARS_FOR_EMBEDDING = 2000  # можно потом подкрутить


def _shorten_for_embedding(text: str, max_chars: int = MAX_CHARS_FOR_EMBEDDING) -> str:
    """
    Урезает слишком длинный текст для эмбеддинга, чтобы не ловить 413 от GigaChat.
    Берём начало и конец, середину выкидываем.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    return head + "\n...\n# [TRUNCATED]\n...\n" + tail

def get_client() -> chromadb.Client:
    """
    Создаёт/открывает клиент Chroma с сохранением индекса в data/index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.Client(
        Settings(
            persist_directory=str(INDEX_DIR),
            is_persistent=True,
        )
    )
    return client


def get_collection():
    """
    Возвращает (или создаёт) коллекцию для кода Reddit.
    """
    client = get_client()
    return client.get_or_create_collection("reddit_code")


def build_index(chunks: List[CodeChunk], batch_size: int = 64):
    """
    Получает список чанков и записывает их в ChromaDB.
    """
    collection = get_collection()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for ch in chunks:
        ids.append(f"{ch.file_path}:{ch.start_line}-{ch.end_line}")
        documents.append(ch.code)
        metadatas.append(
            {
                "file_path": ch.file_path,
                "kind": ch.kind,
                "name": ch.name,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "language": "python",
            }
        )

    total = len(documents)
    print(f"Всего чанков для индексации: {total}")

    for i in range(0, total, batch_size):
        batch_docs_full = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        # Урезаем тексты для эмбеддингов
        batch_docs_short = [_shorten_for_embedding(d) for d in batch_docs_full]

        vectors = embed_texts(batch_docs_short)

        # В Chroma можно сохранять либо короткий вариант, либо полный.
        # Для простоты сейчас кладём короткий (для поиска это ок, есть file_path + строки).
        collection.add(
         ids=batch_ids,
         documents=batch_docs_short,
         metadatas=batch_meta,
          embeddings=vectors,
        )

    print(f"Добавлен батч {i}–{i+len(batch_docs_short)} из {total}")



def search_similar(query: str, k: int = 5):
    """
    Ищет k самых похожих чанков кода по текстовому запросу.
    """
    collection = get_collection()
    query_vec = embed_text(query)

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
    )

    matches = []
    for i in range(len(result["ids"][0])):
        matches.append(
            {
                "id": result["ids"][0][i],
                "document": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result.get("distances", [[None]])[0][i],
            }
        )
    return matches
