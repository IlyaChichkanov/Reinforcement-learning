"""Код результата квиза: сборка, разбор, контрольная сумма.

Формат: W1:IVAN23:BDAACBDABCAD:7F3A
    W1    — неделя курса
    IVAN23 — личный код студента (из roster.csv)
    BDAA… — по одной букве на вопрос, X = без ответа
    7F3A  — контрольная сумма (crc32 от первых трёх полей)

Контрольная сумма защищает от опечаток при копировании, а не от подделки:
соль публичная, и страница квиза целиком открыта студенту. Квиз — это
самопроверка и учёт, а не защищённый экзамен.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

SALT = "rl-course-2026"
LETTERS = "ABCDEFGHIJ"
NO_ANSWER = "X"

CODE_RE = re.compile(r"^W(\d{1,2}):([A-Z0-9]{3,12}):([A-JX]{1,40}):([0-9A-F]{4})$")
STUDENT_RE = re.compile(r"^[A-Z0-9]{3,12}$")


class QuizCodeError(ValueError):
    """Код результата не разобрался: неверный формат или контрольная сумма."""


@dataclass(frozen=True)
class QuizResult:
    week: int
    student: str
    answers: str          # строка из букв, X = вопрос пропущен

    @property
    def n_questions(self) -> int:
        return len(self.answers)

    def indices(self) -> list[int | None]:
        """Номера выбранных вариантов, None для пропущенных вопросов."""
        return [None if ch == NO_ANSWER else LETTERS.index(ch) for ch in self.answers]


def checksum(payload: str) -> str:
    """Четыре hex-символа: младшие 16 бит crc32 от строки с солью."""
    return f"{zlib.crc32((payload + SALT).encode()) & 0xFFFF:04X}"


def encode(week: int, student: str, answers: list[int | None]) -> str:
    """Собрать код результата. answers — номера вариантов, None для пропуска."""
    student = student.strip().upper()
    if not STUDENT_RE.match(student):
        raise QuizCodeError(f"личный код студента должен быть из 3–12 букв и цифр, получено {student!r}")
    letters = "".join(NO_ANSWER if a is None else LETTERS[a] for a in answers)
    payload = f"W{week}:{student}:{letters}"
    return f"{payload}:{checksum(payload)}"


def decode(code: str) -> QuizResult:
    """Разобрать код результата и проверить контрольную сумму."""
    code = code.strip().upper().replace(" ", "")
    match = CODE_RE.match(code)
    if not match:
        raise QuizCodeError(f"не похоже на код результата: {code!r}")
    week, student, letters, crc = match.groups()
    payload = f"W{week}:{student}:{letters}"
    if checksum(payload) != crc:
        raise QuizCodeError("контрольная сумма не сошлась: код скопирован не целиком или изменён")
    return QuizResult(week=int(week), student=student, answers=letters)
