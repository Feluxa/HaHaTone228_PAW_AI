from pathlib import Path
from indexing import extract_chunks_from_python_file

# Берём сам indexing.py как тестовый файл
path = Path("indexing.py")
chunks = extract_chunks_from_python_file(path)

print(f"Найдено чанков: {len(chunks)}")

for ch in chunks[:5]:
    print("--------")
    print(f"KIND: {ch.kind}")
    print(f"NAME: {ch.name}")
    print(f"LINES: {ch.start_line}-{ch.end_line}")
    print("CODE:")
    print(ch.code[:200], "...\n")
