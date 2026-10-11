"""No runtime class may silently shadow an earlier method definition."""

import ast
from collections import Counter
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "plugin_runtime"


def test_runtime_classes_have_no_duplicate_methods():
    duplicates = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef,)):
                continue
            definitions = [
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for name, count in Counter(definitions).items():
                if count > 1:
                    duplicates.append(f"{path.relative_to(RUNTIME_ROOT)}:{node.lineno} {node.name}.{name} ({count})")
    assert not duplicates, "Duplicate runtime class methods:\\n" + "\\n".join(duplicates)
