from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

from .indexing import CodeChunk
from .embeddings import embed_texts, embed_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# --- Ограничения под GigaChat ---
MAX_SHORT_TEXT_CHARS = 850        # < 500 токенов
MAX_SHORT_CODE_LINES = 20         # первые ~20 строк кода


# ============================================================
# Короткий текст для эмбеддингов
# ============================================================

def chunk_to_short_text(chunk: CodeChunk) -> str:
    """
    Создает компактное описание чанка для эмбеддингов.
    Даёт модели структуру: что это, где находится, какая сигнатура.
    """
    code_lines = chunk.code.splitlines()
    preview = "\n".join(code_lines[:MAX_SHORT_CODE_LINES])

    parts = [
        f"TYPE: {chunk.kind}",
        f"NAME: {chunk.name}",
        f"FILE: {chunk.file_path}",
        f"SIGNATURE: {chunk.signature or ''}",
        f"DOCSTRING: {chunk.docstring or ''}",
        "",
        "CODE_PREVIEW:",
        preview,
    ]

    text = "\n".join(parts).strip()

    # Жесткая обрезка — GigaChat иначе даёт 413
    if len(text) > MAX_SHORT_TEXT_CHARS:
        text = text[:MAX_SHORT_TEXT_CHARS] + "\n...[TRUNCATED]..."

    return text


# ============================================================
# ChromaDB
# ============================================================

def get_client() -> chromadb.Client:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.Client(
        Settings(
            persist_directory=str(INDEX_DIR),
            is_persistent=True,
        )
    )


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        "reddit_code",
        metadata={"hnsw:space": "cosine"}
    )


# ============================================================
# Построение индекса
# ============================================================

def build_index(chunks: List[CodeChunk], batch_size: int = 32):
    collection = get_collection()

    print(f"Всего чанков для индексации: {len(chunks)}")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        ids = [
            f"{ch.file_path}:{ch.start_line}-{ch.end_line}"
            for ch in batch
        ]

        short_texts = [chunk_to_short_text(ch) for ch in batch]
        full_texts = [ch.code for ch in batch]

        # Метаданные — только строки и числа
        metadatas = [
            {
                "file_path": ch.file_path,
                "kind": ch.kind,
                "name": ch.name or "",
                "start_line": int(ch.start_line),
                "end_line": int(ch.end_line),
                "signature": ch.signature or "",
                "language": "python",
            }
            for ch in batch
        ]

        # Эмбеддинги считаются ТОЛЬКО по короткому тексту
        vectors = embed_texts(short_texts)

        collection.add(
            ids=ids,
            documents=full_texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

        print(f"→ Indexed {start + len(batch)} / {len(chunks)}")

    print("Индекс успешно построен.")


# ============================================================
# Поиск похожих чанков
# ============================================================

def search_similar(query: str, k: int = 5):
    collection = get_collection()
    qvec = embed_text(query)

    result = collection.query(
        query_embeddings=[qvec],
        n_results=k,
    )

    matches = []
    for i in range(len(result["ids"][0])):
        matches.append(
            {
                "id": result["ids"][0][i],
                "document": result["documents"][0][i],  # полный код
                "metadata": result["metadatas"][0][i],
                "distance": result.get("distances", [[None]])[0][i],
            }
        )

    return matches
