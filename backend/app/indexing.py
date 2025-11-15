from pathlib import Path
from typing import Iterator, List
from dataclasses import dataclass
import ast

# Путь до корня проекта (.../HaHaTone228_PAW_AI)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Путь до локального репозитория reddit
REDDIT_ROOT = PROJECT_ROOT / "data" / "reddit"


def iter_code_files(root: Path, exts=(".py",)) -> Iterator[Path]:
    """
    Итерируемся по всем файлам с нужными расширениями в каталоге root.
    Сейчас работаем только с .py файлами.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in exts:
            yield path


@dataclass
class CodeChunk:
    """
    Один логический кусок кода (функция или класс).
    """
    file_path: str   # относительный путь внутри reddit_repo
    start_line: int
    end_line: int
    kind: str        # "function" или "class"
    name: str        # имя функции/класса
    code: str        # текст кода этого блока


def extract_chunks_from_python_file(path: Path) -> List[CodeChunk]:
    """
    Разбирает .py файл через ast и достает из него функции и классы.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # некоторые файлы могут не парситься — просто пропускаем
        return []

    lines = text.splitlines()
    chunks: List[CodeChunk] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None) or start

            # защита от выхода за пределы
            start = max(start, 1)
            end = min(end, len(lines))

            code_lines = lines[start - 1:end]
            code = "\n".join(code_lines)

            kind = "class" if isinstance(node, ast.ClassDef) else "function"

            chunks.append(
                CodeChunk(
                    file_path=str(path.relative_to(REDDIT_ROOT)),
                    start_line=start,
                    end_line=end,
                    kind=kind,
                    name=node.name,
                    code=code,
                )
            )

    return chunks


def collect_all_chunks(limit_files: int | None = None) -> List[CodeChunk]:
    """
    Проходит по всем .py файлам в reddit_repo и собирает чанки.
    limit_files — можно ограничить количество файлов для отладки.
    """
    if not REDDIT_ROOT.exists():
        print(f"reddit_repo not found: {REDDIT_ROOT}")
        return []

    all_chunks: List[CodeChunk] = []
    total_files = 0
    files_with_chunks = 0

    for i, path in enumerate(iter_code_files(REDDIT_ROOT)):
        if limit_files is not None and i >= limit_files:
            break

        total_files += 1
        chunks = extract_chunks_from_python_file(path)
        if chunks:
            files_with_chunks += 1
        all_chunks.extend(chunks)

    print(f"Всего файлов .py обработано: {total_files}")
    print(f"Файлов, где были найдены функции/классы: {files_with_chunks}")
    print(f"Всего чанков (функций/классов): {len(all_chunks)}")

    return all_chunks


def debug_one_file_with_chunks():
    """
    Найти первый .py файл, в котором реально есть чанки, и вывести несколько.
    """
    if not REDDIT_ROOT.exists():
        print(f"reddit_repo not found: {REDDIT_ROOT}")
        return

    for path in iter_code_files(REDDIT_ROOT):
        chunks = extract_chunks_from_python_file(path)
        if not chunks:
            continue

        print(f"FILE: {path}")
        print(f"Найдено чанков: {len(chunks)}\n")

        for ch in chunks[:5]:
            print("-" * 60)
            print(f"{ch.kind.upper()} {ch.name} ({ch.start_line}-{ch.end_line})")
            print(ch.code)
            print()
        break
    else:
        print("Вообще не нашли ни одного файла с функциями/классами 🤔")




def build_reddit_index():
    """
    Собрать все python-чанки и записать их в векторную базу.
    """
    #локальный импорт делаем
    from .vector_store import build_index  # импорт в конце, чтобы не было циклов

    chunks = collect_all_chunks()
    if not chunks:
        print("Чанки не найдены, индекс не строим")
        return

    print("Начинаем построение векторного индекса...")
    build_index(chunks)
    print("Индекс построен.")


if __name__ == "__main__":
    build_reddit_index()

    #Для отладки потом можно вернуть:
    # collect_all_chunks()
    # debug_one_file_with_chunks()

    # Для общей статистики по всем файлам:
    #collect_all_chunks()

    # Для детального просмотра первых чанков из какого-нибудь файла:
    # debug_one_file_with_chunks()