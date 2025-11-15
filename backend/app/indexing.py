from pathlib import Path
from typing import Iterator, List, Optional
from dataclasses import dataclass

from tree_sitter import Parser
from tree_sitter_languages import get_language


# ============================================================
# Пути
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REDDIT_ROOT = PROJECT_ROOT / "data" / "reddit"

# ============================================================
# Tree-sitter init
# ============================================================

PY_LANG = get_language("python")
parser = Parser()
parser.set_language(PY_LANG)

# ============================================================
# Dataclass — единый формат чанка
# ============================================================

@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    kind: str              # file | class | function | method
    name: str
    signature: Optional[str]
    docstring: Optional[str]
    code: str


# ============================================================
# Утилиты
# ============================================================

def try_relative_path(path: Path) -> str:
    """Безопасный relative_to — если файл не в reddit root, возвращаем абсолютный путь."""
    try:
        return str(path.relative_to(REDDIT_ROOT))
    except ValueError:
        return str(path)


def iter_code_files(root: Path, exts=(".py",)) -> Iterator[Path]:
    """Итерируем python-файлы."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in exts:
            yield path


def node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def get_docstring_from_body(source: bytes, node) -> Optional[str]:
    """Ищем докстринг — первый string_literal внутри блока."""
    for child in node.children:
        if child.type == "block":
            for grand in child.children:
                if grand.type == "expression_statement" and len(grand.children):
                    literal = grand.children[0]
                    if literal.type == "string":
                        return node_text(source, literal)
    return None


def get_signature_from_function(source: bytes, node) -> str:
    """Вытягиваем строчку def ...(...):"""
    text = node_text(source, node)
    return text.split(":")[0].strip()


# ============================================================
# Основная функция — вытаскивание чанков из одного файла
# ============================================================

def extract_chunks_from_python_file(path: Path) -> List[CodeChunk]:
    source = path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    chunks: List[CodeChunk] = []

    # --------------------------------------------------------
    # FILE-level chunk
    # --------------------------------------------------------
    file_code = source.decode("utf-8", errors="ignore")

    chunks.append(
        CodeChunk(
            file_path=try_relative_path(path),
            start_line=1,
            end_line=file_code.count("\n") + 1,
            kind="file",
            name=path.name,
            signature=None,
            docstring=None,
            code=file_code,
        )
    )

    # --------------------------------------------------------
    # Рекурсивный обход
    # --------------------------------------------------------
    def walk(node, parent_kind=None):
        for child in node.children:

            # === CLASS ===
            if child.type == "class_definition":
                name = ""
                for c in child.children:
                    if c.type == "identifier":
                        name = node_text(source, c)
                        break

                doc = get_docstring_from_body(source, child)
                code = node_text(source, child)
                start = child.start_point[0] + 1
                end = child.end_point[0] + 1

                chunks.append(
                    CodeChunk(
                        file_path=try_relative_path(path),
                        start_line=start,
                        end_line=end,
                        kind="class",
                        name=name,
                        signature=f"class {name}",
                        docstring=doc,
                        code=code,
                    )
                )

                walk(child, parent_kind="class")

            # === FUNCTION / METHOD ===
            elif child.type == "function_definition":
                name = ""
                for c in child.children:
                    if c.type == "identifier":
                        name = node_text(source, c)
                        break

                doc = get_docstring_from_body(source, child)
                signature = get_signature_from_function(source, child)
                code = node_text(source, child)
                start = child.start_point[0] + 1
                end = child.end_point[0] + 1

                kind = "method" if parent_kind == "class" else "function"

                chunks.append(
                    CodeChunk(
                        file_path=try_relative_path(path),
                        start_line=start,
                        end_line=end,
                        kind=kind,
                        name=name,
                        signature=signature,
                        docstring=doc,
                        code=code,
                    )
                )

                walk(child, parent_kind=kind)

            else:
                walk(child, parent_kind=parent_kind)

    walk(root)
    return chunks


# ============================================================
# Сбор всех чанков
# ============================================================

def collect_all_chunks(limit_files: int | None = None) -> List[CodeChunk]:
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
    print(f"С файлами, где есть чанки: {files_with_chunks}")
    print(f"Всего чанков: {len(all_chunks)}")

    return all_chunks


# ============================================================
# Debug
# ============================================================

def debug_one_file_with_chunks():
    for path in iter_code_files(REDDIT_ROOT):
        chunks = extract_chunks_from_python_file(path)
        if not chunks:
            continue

        print(f"FILE: {path}")
        print(f"Найдено чанков: {len(chunks)}")

        for ch in chunks[:5]:
            print("-" * 60)
            print(f"{ch.kind.upper()} {ch.name} ({ch.start_line}-{ch.end_line})")
            print(ch.code[:200], "...\n")
        break


# ============================================================
# Build index
# ============================================================

def build_reddit_index():
    from .vector_store import build_index

    chunks = collect_all_chunks()
    if not chunks:
        print("Чанки не найдены, индекс не строим")
        return

    print("Начинаем построение векторного индекса...")
    build_index(chunks)
    print("Индекс построен.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_reddit_index()
