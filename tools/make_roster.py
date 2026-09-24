"""Личные коды студентов из списка группы.

На вход CSV со столбцами «фамилия,имя,группа» (заголовок обязателен, порядок
столбцов любой, лишние столбцы игнорируются). На выход roster.csv с личным
кодом каждого студента — его студент вводит на странице квиза.

    uv run python tools/make_roster.py students.csv
    uv run python tools/make_roster.py students.csv --out roster.csv

Коды детерминированы: тот же список даёт те же коды, поэтому файл можно
пересоздавать, не ломая уже сданные работы.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

# Транслитерация для кода: только буквы, которые попадут в код.
TRANSLIT = {
    "а": "A", "б": "B", "в": "V", "г": "G", "д": "D", "е": "E", "ё": "E",
    "ж": "Z", "з": "Z", "и": "I", "й": "I", "к": "K", "л": "L", "м": "M",
    "н": "N", "о": "O", "п": "P", "р": "R", "с": "S", "т": "T", "у": "U",
    "ф": "F", "х": "H", "ц": "C", "ч": "C", "ш": "S", "щ": "S", "ъ": "",
    "ы": "Y", "ь": "", "э": "E", "ю": "U", "я": "A",
}


def translit(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch.isascii() and ch.isalpha():
            out.append(ch.upper())
    return "".join(out)


def student_code(surname: str, name: str, group: str, bump: int = 0) -> str:
    """Четыре буквы фамилии плюс две цифры от хеша всей записи."""
    stem = (translit(surname) + "XXXX")[:4]
    seed = f"{surname.strip().lower()}|{name.strip().lower()}|{group.strip().lower()}|{bump}"
    digits = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 100
    return f"{stem}{digits:02d}"


def read_students(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: пустой файл")
    fields = {(k or "").strip().lower(): k for k in rows[0]}
    required = ("фамилия", "имя", "группа")
    missing = [c for c in required if c not in fields]
    if missing:
        sys.exit(f"{path}: нет столбцов {', '.join(missing)}; есть {', '.join(fields)}")
    return [{c: (row[fields[c]] or "").strip() for c in required} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Личные коды студентов для квизов")
    parser.add_argument("students", type=Path, help="CSV со столбцами фамилия,имя,группа")
    parser.add_argument("--out", type=Path, default=Path("roster.csv"), help="куда писать (по умолчанию roster.csv)")
    args = parser.parse_args()

    students = read_students(args.students)
    seen: dict[str, dict[str, str]] = {}
    roster = []
    for s in students:
        bump = 0
        code = student_code(s["фамилия"], s["имя"], s["группа"])
        while code in seen:                       # коллизия: тот же код у другого человека
            bump += 1
            code = student_code(s["фамилия"], s["имя"], s["группа"], bump)
        seen[code] = s
        roster.append({"код": code, **s})

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["код", "фамилия", "имя", "группа"])
        writer.writeheader()
        writer.writerows(roster)

    print(f"{len(roster)} студентов -> {args.out}\n")
    for r in roster:
        print(f"  {r['код']:8s} {r['фамилия']} {r['имя']} ({r['группа']})")
    print("\nРаздайте каждому его код: он вводит его на странице квиза в зачётном режиме.")
    print(f"Файл {args.out} держите вне публичного репозитория (он уже в .gitignore).")


if __name__ == "__main__":
    main()
