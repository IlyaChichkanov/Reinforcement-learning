"""Проверка кода результата квиза: формат и контрольная сумма.

Правильных ответов не знает и баллов не считает — только говорит, что код
скопирован целиком и не испорчен. Запускается в репозитории студента из
GitHub Actions, поэтому ключ ответов ему не нужен.

    uv run python tools/check_quiz_code.py quiz/week01.txt
    uv run python tools/check_quiz_code.py quiz/*.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quizcode import QuizCodeError, decode  # noqa: E402

PLACEHOLDER = "вставьте сюда код результата"


def read_code(path: Path) -> str | None:
    """Первая непустая строка файла, не считая комментариев и заготовки."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and PLACEHOLDER not in line.lower():
            return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить код результата квиза")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--required", action="store_true", help="считать ошибкой пустой файл")
    args = parser.parse_args()

    failed = False
    for path in args.files:
        if not path.exists():
            print(f"{path}: файла нет")
            failed = failed or args.required
            continue
        code = read_code(path)
        if code is None:
            print(f"{path}: код не вставлен")
            failed = failed or args.required
            continue
        try:
            result = decode(code)
        except QuizCodeError as e:
            print(f"{path}: {e}")
            failed = True
            continue
        answered = sum(ch != "X" for ch in result.answers)
        print(f"{path}: неделя {result.week}, студент {result.student}, "
              f"отвечено {answered} из {result.n_questions} — код в порядке")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
