"""Личные коды студентов из списка группы.

На вход CSV со столбцами «фамилия,имя,группа» (заголовок обязателен, порядок
столбцов любой, лишние столбцы игнорируются). На выход roster.csv с личным
кодом каждого студента — его студент вводит на странице квиза.

    uv run python tools/make_roster.py students.csv
    uv run python tools/make_roster.py students.csv --out roster.csv

Вместо отдельных столбцов можно дать таблицу логинов (столбец «ФИО», как в
course-admin/students.csv) и одну группу на всех:

    uv run python tools/make_roster.py students.csv --from-fio --group RL-2026

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


def read_fio(path: Path, group: str) -> list[dict[str, str]]:
    """Студенты из таблицы со столбцом «ФИО» (над заголовком могут быть строки-пояснения)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    at = next((i for i, r in enumerate(rows) if any((c or "").strip().lower() == "фио" for c in r)), None)
    if at is None:
        sys.exit(f"{path}: нет столбца «ФИО»")
    i_fio = [(c or "").strip().lower() for c in rows[at]].index("фио")
    students = []
    for r in rows[at + 1:]:
        parts = (r[i_fio] if len(r) > i_fio else "").split()
        if len(parts) >= 2:
            students.append({"фамилия": parts[0], "имя": parts[1], "группа": group})
    return students


def build_roster(students: list[dict[str, str]]) -> list[dict[str, str]]:
    """Личный код каждому; при совпадении кодов у разных людей — следующий вариант хеша."""
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
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description="Личные коды студентов для квизов")
    parser.add_argument("students", type=Path, help="CSV со столбцами фамилия,имя,группа (или ФИО с --from-fio)")
    parser.add_argument("--from-fio", action="store_true", help="брать фамилию и имя из столбца «ФИО»")
    parser.add_argument("--group", default="", help="группа для всех (с --from-fio)")
    parser.add_argument("--out", type=Path, default=Path("roster.csv"), help="куда писать (по умолчанию roster.csv)")
    args = parser.parse_args()

    students = read_fio(args.students, args.group) if args.from_fio else read_students(args.students)
    roster = build_roster(students)

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
