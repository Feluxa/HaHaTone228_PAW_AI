from pathlib import Path
from typing import Iterator, List
from dataclasses import dataclass
import ast

# Путь до корня проекта (.../HaHaTone228_PAW_AI)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Путь до локального репозитория reddit
REDDIT_ROOT = PROJECT_ROOT / "data" / "reddit"

TEXT_EXTS = (".json", ".md", ".txt", ".ini", ".cfg", ".yaml", ".yml")




@dataclass
class CodeChunk:
    """
    Один логический кусок кода (функция или класс).
    """
    file_path: str   # относительный путь внутри reddit_repo
    start_line: int
    end_line: int
    kind: str        # "function" / "class" / "text" / "config" / "doc"
    name: str        # имя функции/класса или имя файла
    code: str        # текст кода/документа
    language: str = "python"  # "python", "json", "markdown", "text", "config", ...

def iter_code_files(root: Path, exts=(".py",)) -> Iterator[Path]:
    """
    Итерируемся по всем файлам с нужными расширениями в каталоге root.
    Сейчас работаем только с .py файлами.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in exts:
            yield path


def iter_text_files(root: Path, exts=TEXT_EXTS) -> Iterator[Path]:
    """
    Итерируемся по всем текстовым файлам (json, md, txt, конфиги).
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path

def _detect_language_from_suffix(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".json":
        return "json"
    if suf in (".md", ".markdown"):
        return "markdown"
    if suf in (".yaml", ".yml"):
        return "yaml"
    if suf in (".ini", ".cfg"):
        return "config"
    if suf == ".txt":
        return "text"
    return "text"


def extract_chunks_from_text_file(path: Path) -> List[CodeChunk]:
    """
    Достаёт чанки из текстовых файлов (json, md, txt, конфиги).
    Для простоты: один чанк = весь файл.
    Если файл очень большой, при эмбеддинге мы всё равно его урежем.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = text.splitlines()
    language = _detect_language_from_suffix(path)
    rel_path = str(path.relative_to(REDDIT_ROOT))

    chunk = CodeChunk(
        file_path=rel_path,
        start_line=1,
        end_line=len(lines) if lines else 1,
        kind="text",
        name=path.name,
        code=text,
        language=language,
    )

    return [chunk]


def extract_chunks_from_python_file(path: Path) -> List[CodeChunk]:
    """
    Разбирает .py файл через ast и достает из него функции и классы.
    В текст чанка добавляем:
    - имя функции/класса,
    - docstring (если есть),
    - сам код.
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

            # docstring, если есть
            docstring = ast.get_docstring(node)
            header_parts = [f"# {kind} {node.name}"]
            if docstring:
                header_parts.append(f"# doc: {docstring.strip().replace('\n', ' ')}")
            header_text = "\n".join(header_parts)

            full_text = f"{header_text}\n\n{code}"

            chunks.append(
                CodeChunk(
                    file_path=str(path.relative_to(REDDIT_ROOT)),
                    start_line=start,
                    end_line=end,
                    kind=kind,
                    name=node.name,
                    code=full_text,
                    language="python",
                )
            )

    return chunks


def collect_all_chunks(limit_files: int | None = None) -> List[CodeChunk]:
    """
    Проходит по всем .py файлам в reddit_repo и собирает чанки,
    а также по текстовым файлам (json, md, txt, конфиги).
    limit_files — можно ограничить количество файлов для отладки (по коду и по тексту отдельно).
    """
    if not REDDIT_ROOT.exists():
        print(f"reddit_repo not found: {REDDIT_ROOT}")
        return []

    all_chunks: List[CodeChunk] = []
    total_py_files = 0
    py_files_with_chunks = 0

    # 1) Python-файлы
    for i, path in enumerate(iter_code_files(REDDIT_ROOT)):
        if limit_files is not None and i >= limit_files:
            break

        total_py_files += 1
        chunks = extract_chunks_from_python_file(path)
        if chunks:
            py_files_with_chunks += 1
        all_chunks.extend(chunks)

    print(f"Всего файлов .py обработано: {total_py_files}")
    print(f"Файлов, где были найдены функции/классы: {py_files_with_chunks}")
    print(f"Всего чанков (функций/классов): {len(all_chunks)}")

    # 2) Текстовые файлы (json, md, txt, конфиги)
    total_text_files = 0
    text_files_with_chunks = 0

    for j, path in enumerate(iter_text_files(REDDIT_ROOT)):
        if limit_files is not None and j >= limit_files:
            break

        total_text_files += 1
        chunks = extract_chunks_from_text_file(path)
        if chunks:
            text_files_with_chunks += 1
        all_chunks.extend(chunks)

    print(f"Всего текстовых файлов (json/md/txt/config) обработано: {total_text_files}")
    print(f"Файлов, где были найдены текстовые чанки: {text_files_with_chunks}")
    print(f"ИТОГО ВСЕХ ЧАНКОВ (код + текст): {len(all_chunks)}")

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