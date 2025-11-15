from tree_sitter import Parser
from tree_sitter_languages import get_language

try:
    lang = get_language("python")
    parser = Parser()
    parser.set_language(lang)
    print("✓ Tree-sitter: Python grammar loaded successfully!")
except Exception as e:
    print("✗ Tree-sitter failed:", e)
