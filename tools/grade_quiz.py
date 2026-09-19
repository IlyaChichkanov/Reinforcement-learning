"""Журнал результатов квизов из кодов, которые сдали студенты.

На вход — файлы с кодами результатов (по одному в строке; строки с # и пустые
пропускаются) или каталог, который обходится рекурсивно. На выход — таблица
CSV: кто, какая неделя, сколько баллов, что ответил по каждому вопросу.

    uv run python tools/grade_quiz.py collected/ --out journal.csv
    uv run python tools/grade_quiz.py codes.txt --roster roster.csv

Ключ ответов берётся из quizzes/week*.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quizcode import LETTERS, QuizCodeError, decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUIZ_DIR = ROOT / "quizzes"


def load_keys() -> dict[int, dict]:
    """{неделя: {"answers": [индексы правильных], "ids": [...], "pass": N}}."""
    keys = {}
    for path in sorted(QUIZ_DIR.glob("week*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        keys[data["week"]] = {
            "answers": [q["answer"] for q in data["questions"]],
            "ids": [q["id"] for q in data["questions"]],
            "pass": data.get("pass_score", 0),
            "title": data["title"],
        }
    if not keys:
        sys.exit(f"в {QUIZ_DIR} нет файлов week*.json")
    return keys


def load_roster(path: Path | None) -> dict[str, str]:
    """{личный код: «Фамилия Имя (группа)»}."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            (row["код"] or "").strip().upper():
                f"{row['фамилия']} {row['имя']} ({row['группа']})".strip()
            for row in csv.DictReader(f)
        }


def iter_codes(paths: list[Path]):
    """(файл, строка) для каждой осмысленной строки во всех переданных путях."""
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if file.suffix.lower() not in {".txt", ".csv", ".md", ""}:
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "вставьте" not in line.lower():
                    yield file, line


def main() -> int:
    parser = argparse.ArgumentParser(description="Журнал результатов квизов")
    parser.add_argument("paths", nargs="+", type=Path, help="файлы с кодами или каталоги")
    parser.add_argument("--out", type=Path, default=Path("journal.csv"))
    parser.add_argument("--roster", type=Path, default=Path("roster.csv"))
    args = parser.parse_args()

    keys = load_keys()
    roster = load_roster(args.roster)
    rows, problems = [], []
    seen: set[tuple[str, int]] = set()

    for file, line in iter_codes(args.paths):
        try:
            result = decode(line)
        except QuizCodeError as e:
            problems.append(f"{file}: {e}")
            continue
        key = keys.get(result.week)
        if key is None:
            problems.append(f"{file}: нет ключа для недели {result.week}")
            continue
        expected = len(key["answers"])
        if result.n_questions != expected:
            problems.append(f"{file}: в коде {result.n_questions} ответов, в квизе недели {result.week} их {expected}")
            continue
        if (result.student, result.week) in seen:
            problems.append(f"{file}: повторный код для {result.student}, неделя {result.week} — взят первый")
            continue
        seen.add((result.student, result.week))

        given = result.indices()
        correct = [g is not None and g == right for g, right in zip(given, key["answers"])]
        row = {
            "код": result.student,
            "студент": roster.get(result.student, ""),
            "неделя": result.week,
            "балл": sum(correct),
            "из": expected,
            "зачёт": "да" if sum(correct) >= key["pass"] else "нет",
            "ответы": result.answers,
        }
        for qid, ok in zip(key["ids"], correct):
            row[qid] = 1 if ok else 0
        rows.append(row)

    if not rows:
        print("Ни одного корректного кода не найдено.")
        for p in problems:
            print("  ", p)
        return 1

    rows.sort(key=lambda r: (r["неделя"], r["студент"] or r["код"]))
    fieldnames = ["код", "студент", "неделя", "балл", "из", "зачёт", "ответы"]
    fieldnames += [qid for week in sorted(keys) for qid in keys[week]["ids"]]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} работ -> {args.out}\n")
    for r in rows:
        who = r["студент"] or r["код"]
        print(f"  неделя {r['неделя']}  {who:38.38s} {r['балл']:2d} из {r['из']}  зачёт: {r['зачёт']}")

    by_week: dict[int, list[int]] = {}
    for r in rows:
        by_week.setdefault(r["неделя"], []).append(r["балл"])
    print()
    for week, scores in sorted(by_week.items()):
        print(f"  неделя {week}: работ {len(scores)}, средний балл {sum(scores) / len(scores):.1f} из {len(keys[week]['answers'])}")

    # Какие вопросы оказались самыми трудными — что стоит разобрать на семинаре.
    for week in sorted(by_week):
        ids = keys[week]["ids"]
        week_rows = [r for r in rows if r["неделя"] == week]
        hard = sorted(((sum(r[q] for r in week_rows) / len(week_rows), q) for q in ids))[:3]
        print(f"  неделя {week}, труднее всего: " + ", ".join(f"{q} ({share:.0%})" for share, q in hard))

    if problems:
        print("\nНе разобрано:")
        for p in problems:
            print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
