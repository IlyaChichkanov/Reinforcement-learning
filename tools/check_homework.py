"""Проверка сданного ноутбука: всё ли выполнено и не осталось ли заготовок.

Правильность решения не проверяет — это делают `assert` внутри самого
ноутбука и преподаватель. Здесь только то, что студент часто забывает:
запустить ноутбук целиком, убрать `raise NotImplementedError`, не оставить
ячейку с ошибкой.

    uv run python tools/check_homework.py homework.ipynb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEFTOVERS = ("raise NotImplementedError", "# TODO: ваш код здесь", "TODO: ваш код")


def source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def check(path: Path) -> tuple[list[str], list[str]]:
    """Возвращает (проблемы, замечания)."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"{path}: не читается как ноутбук ({e})"], []

    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not cells:
        return [f"{path}: в ноутбуке нет ячеек с кодом"], []

    problems, notes = [], []

    not_run = [i for i, c in enumerate(cells) if c.get("execution_count") is None and source(c).strip()]
    if len(not_run) == len(cells):
        problems.append(f"{path}: ноутбук не запускался — выполните «Restart & Run All» и сохраните")
    elif not_run:
        problems.append(f"{path}: не выполнено ячеек с кодом: {len(not_run)} из {len(cells)}")

    failed = [i for i, c in enumerate(cells)
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if failed:
        names = ", ".join(str(i + 1) for i in failed[:6])
        problems.append(f"{path}: ячейки с ошибкой: {names}{' и другие' if len(failed) > 6 else ''}")

    left = [i for i, c in enumerate(cells) if any(m in source(c) for m in LEFTOVERS)]
    if left:
        problems.append(f"{path}: остались незаполненные заготовки в {len(left)} ячейках")

    todo_md = sum(1 for c in nb.get("cells", [])
                  if c.get("cell_type") == "markdown" and "TODO" in source(c))
    if todo_md:
        notes.append(f"{path}: текстовых ответов с пометкой TODO: {todo_md} — их проверяет преподаватель вручную")

    return problems, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить сданный ноутбук")
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()

    all_problems, all_notes = [], []
    for path in args.notebooks:
        if not path.exists():
            all_problems.append(f"{path}: файла нет")
            continue
        problems, notes = check(path)
        all_problems += problems
        all_notes += notes

    for note in all_notes:
        print("замечание:", note)
    if all_problems:
        print("\nНадо поправить:")
        for p in all_problems:
            print("  -", p)
        return 1
    print("\nНоутбук выполнен целиком, заготовок и ячеек с ошибками не осталось.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
