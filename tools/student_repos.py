"""Личные репозитории студентов в организации курса: создать, проверить, собрать работы.

Замена GitHub Classroom (закрыт 28.08.2026). Каждому студенту — закрытый репозиторий
``<org>/<prefix>-<логин>`` из шаблона ``<org>/<prefix>``; студента приглашают в него
с правом записи. Список берётся из CSV, выгруженного из таблицы логинов: нужны
столбцы с «ФИО» и «логин» в названии (как «№,ФИО,GitHub-логин»), пустые логины
пропускаются.

    python tools/student_repos.py check   --students students.csv
    python tools/student_repos.py create  --students students.csv --org rl-course-misis --prefix rl-hw-block1
    python tools/student_repos.py status  --students students.csv --org rl-course-misis --prefix rl-hw-block1
    python tools/student_repos.py collect --students students.csv --org rl-course-misis --prefix rl-hw-block1 \\
                                          --into block1 [--before 2026-10-28T23:59:59+03:00]

Токен — из GITHUB_TOKEN: fine-grained, на организацию, права Administration и Contents
(запись), Metadata (чтение). Адрес API можно подменить через GITHUB_API_URL (для тестов).
``create`` можно запускать сколько угодно раз: существующие репозитории и уже
приглашённые студенты пропускаются.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
RE_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
FILES = ("homework.ipynb",)     # что забирать из корня репозитория студента
QUIZ_DIR = "quiz"               # и все .txt из этого каталога


@dataclass
class Student:
    name: str
    login: str      # нормализованный, как вписан
    raw: str        # как вписан в таблицу — для сообщений

    def repo(self, prefix: str) -> str:
        return f"{prefix}-{self.login.lower()}"


# ---------- список студентов ----------

def normalize_login(text: str) -> str | None:
    """«@Ivan», «github.com/Ivan/», « Ivan » -> «Ivan»; мусор -> None."""
    s = text.strip().strip("@").strip()
    m = re.search(r"github\.com/([^/?#\s]+)", s)
    if m:
        s = m.group(1)
    s = s.strip("/").strip()
    return s if RE_LOGIN.match(s) else None


def read_students(path: Path) -> tuple[list[Student], list[str]]:
    """(студенты с корректным логином, сообщения о строках с плохим логином)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    # над таблицей могут быть название и инструкция — ищем строку, где есть оба столбца
    header_at = next((i for i, r in enumerate(rows)
                      if any("логин" in c.lower() for c in r) and any("фио" in c.lower() for c in r)), None)
    if header_at is None:
        sys.exit(f"{path}: не нашёл строку заголовка со столбцами «ФИО» и «...логин...»")
    header = [c.strip().lower() for c in rows[header_at]]
    i_login = next(i for i, c in enumerate(header) if "логин" in c)
    i_name = next(i for i, c in enumerate(header) if "фио" in c)

    students, bad = [], []
    seen: dict[str, str] = {}
    for r in rows[header_at + 1:]:
        if len(r) <= max(i_login, i_name):
            r = r + [""] * (max(i_login, i_name) + 1 - len(r))
        name, raw = r[i_name].strip(), r[i_login].strip()
        if not name or not raw:
            continue
        login = normalize_login(raw)
        if login is None:
            bad.append(f"{name}: «{raw}» не похоже на логин GitHub")
            continue
        if login.lower() in seen:
            bad.append(f"{name}: логин {login} уже вписан у «{seen[login.lower()]}»")
            continue
        seen[login.lower()] = name
        students.append(Student(name, login, raw))
    return students, bad


# ---------- GitHub API ----------

class GitHub:
    def __init__(self, token: str | None, dry_run: bool = False):
        if not token and not dry_run:
            sys.exit("нет токена: задайте переменную окружения GITHUB_TOKEN")
        self.token = token
        self.dry_run = dry_run

    def call(self, method: str, path: str, body: dict | None = None,
             accept: str = "application/vnd.github+json") -> tuple[int, object]:
        url = path if path.startswith("http") else f"{API}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", accept)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    status = resp.status
                break
            except urllib.error.HTTPError as e:
                raw, status = e.read(), e.code
                retry = status in (502, 503, 504) or (
                    status in (403, 429) and e.headers.get("Retry-After"))
                if retry and attempt < 3:
                    time.sleep(int(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                break
        if accept.endswith("raw") or accept.endswith("raw+json"):
            return status, raw
        try:
            return status, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return status, raw.decode(errors="replace")

    def get(self, path: str, **kw):
        return self.call("GET", path, **kw)


def fail(what: str, status: int, payload) -> str:
    msg = payload.get("message") if isinstance(payload, dict) else payload
    return f"{what}: HTTP {status} {msg}"


# ---------- команды ----------

def cmd_check(gh: GitHub, students: list[Student], args) -> int:
    missing = 0
    for s in students:
        status, data = gh.get(f"/users/{s.login}")
        if status == 200:
            note = "" if data.get("login") == s.login else f" (на GitHub: {data.get('login')})"
            print(f"  ok        {s.login:24} {s.name}{note}")
        elif status == 404:
            missing += 1
            print(f"  НЕ НАЙДЕН {s.raw:24} {s.name}")
        else:
            missing += 1
            print(f"  ошибка    {s.raw:24} {s.name}: {fail('', status, data)}")
    print(f"\nНайдено {len(students) - missing} из {len(students)}.")
    return 1 if missing else 0


def cmd_create(gh: GitHub, students: list[Student], args) -> int:
    template = args.template or f"{args.org}/{args.prefix}"
    if not gh.dry_run:
        status, data = gh.get(f"/repos/{template}")
        if status != 200:
            sys.exit(fail(f"шаблон {template} недоступен", status, data))
        if not data.get("is_template"):
            sys.exit(f"{template} не отмечен как Template repository (Settings → General)")

    created = existing = invited = errors = 0
    for s in students:
        repo = f"{args.org}/{s.repo(args.prefix)}"
        if gh.dry_run:
            print(f"  [dry-run] {repo} ← {template}; пригласить {s.login} ({s.name})")
            continue

        status, data = gh.get(f"/users/{s.login}")
        if status != 200:
            errors += 1
            print(f"  ПРОПУЩЕН  {s.name}: логин «{s.raw}» не найден на GitHub")
            continue

        status, data = gh.get(f"/repos/{repo}")
        if status == 404:
            status, data = gh.call("POST", f"/repos/{template}/generate", {
                "owner": args.org, "name": s.repo(args.prefix), "private": True,
                "description": f"{args.prefix}: {s.name}"})
            if status != 201:
                errors += 1
                print("  ОШИБКА    " + fail(f"{repo} не создан", status, data))
                continue
            created += 1
            what = "создан"
        elif status == 200:
            existing += 1
            what = "уже был"
        else:
            errors += 1
            print("  ОШИБКА    " + fail(repo, status, data))
            continue

        # репозиторий из шаблона появляется не мгновенно — несколько попыток
        for attempt in range(5):
            status, data = gh.call("PUT", f"/repos/{repo}/collaborators/{s.login}",
                                   {"permission": "push"})
            if status != 404:
                break
            time.sleep(2 + 2 * attempt)
        if status == 201:
            invited += 1
            print(f"  {what:9} {repo}; приглашение отправлено {s.login}")
        elif status == 204:
            print(f"  {what:9} {repo}; {s.login} уже участник")
        else:
            errors += 1
            print("  ОШИБКА    " + fail(f"{repo}: приглашение {s.login}", status, data))

    if not gh.dry_run:
        print(f"\nСоздано {created}, уже было {existing}, приглашений {invited}, ошибок {errors}.")
    return 1 if errors else 0


def student_commits(gh: GitHub, repo: str, login: str) -> list[dict]:
    status, data = gh.get(f"/repos/{repo}/commits?per_page=100")
    if status != 200 or not isinstance(data, list):
        return []
    return [c for c in data if ((c.get("author") or {}).get("login") or "").lower() == login.lower()]


def cmd_status(gh: GitHub, students: list[Student], args) -> int:
    print(f"  {'логин':24} {'приглашение':12} {'коммитов':9} последний  ФИО")
    for s in students:
        repo = f"{args.org}/{s.repo(args.prefix)}"
        status, _ = gh.get(f"/repos/{repo}")
        if status != 200:
            print(f"  {s.login:24} {'нет репо':12} {'':9} {'':10} {s.name}")
            continue
        _, inv = gh.get(f"/repos/{repo}/invitations")
        pending = any(((i.get("invitee") or {}).get("login") or "").lower() == s.login.lower()
                      for i in (inv if isinstance(inv, list) else []))
        commits = student_commits(gh, repo, s.login)
        last = commits[0]["commit"]["committer"]["date"][:10] if commits else ""
        print(f"  {s.login:24} {'не принято' if pending else 'принято':12} "
              f"{len(commits):<9} {last:10} {s.name}")
    return 0


def fetch(gh: GitHub, repo: str, path: str, ref: str | None) -> bytes | None:
    q = f"?ref={urllib.parse.quote(ref)}" if ref else ""
    status, data = gh.get(f"/repos/{repo}/contents/{urllib.parse.quote(path)}{q}",
                          accept="application/vnd.github.raw+json")
    return data if status == 200 else None


def write_if_changed(path: Path, content: bytes) -> bool:
    if path.exists() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def cmd_collect(gh: GitHub, students: list[Student], args) -> int:
    into = Path(args.into)
    changed = 0
    for s in students:
        repo = f"{args.org}/{s.repo(args.prefix)}"
        ref = None
        if args.before:
            until = urllib.parse.quote(args.before)
            status, data = gh.get(f"/repos/{repo}/commits?until={until}&per_page=1")
            if status != 200 or not data:
                print(f"  {s.login:24} нет коммитов до {args.before}")
                continue
            ref = data[0]["sha"]
        else:
            status, data = gh.get(f"/repos/{repo}/commits?per_page=1")
            if status != 200 or not data:
                print(f"  {s.login:24} нет репозитория")
                continue
            ref = data[0]["sha"]
        commit = data[0]

        files: dict[str, bytes] = {}
        for name in FILES:
            content = fetch(gh, repo, name, ref)
            if content is not None:
                files[name] = content
        status, listing = gh.get(f"/repos/{repo}/contents/{QUIZ_DIR}?ref={ref}")
        for item in listing if status == 200 and isinstance(listing, list) else []:
            if item.get("type") == "file" and item["name"].endswith(".txt"):
                content = fetch(gh, repo, f"{QUIZ_DIR}/{item['name']}", ref)
                if content is not None:
                    files[f"{QUIZ_DIR}/{item['name']}"] = content

        meta = {"login": s.login, "name": s.name, "repo": repo, "sha": ref,
                "date": commit["commit"]["committer"]["date"],
                "by_student": ((commit.get("author") or {}).get("login") or "").lower() == s.login.lower()}
        files["meta.json"] = (json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode()

        dest = into / s.login.lower()
        n = sum(write_if_changed(dest / rel, content) for rel, content in files.items())
        changed += n
        print(f"  {s.login:24} {meta['date'][:16]} {ref[:7]} {'обновлено' if n else 'без изменений'}")
    print(f"\nИзменено файлов: {changed}.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Личные репозитории студентов в организации курса")
    p.add_argument("command", choices=["check", "create", "status", "collect"])
    p.add_argument("--students", type=Path, required=True, help="CSV с ФИО и GitHub-логинами")
    p.add_argument("--org", help="организация, например rl-course-misis")
    p.add_argument("--prefix", help="префикс репозиториев = имя шаблона, например rl-hw-block1")
    p.add_argument("--template", help="шаблон, по умолчанию <org>/<prefix>")
    p.add_argument("--into", help="collect: куда складывать работы")
    p.add_argument("--before", help="collect: взять последний коммит не позже этого момента (ISO 8601)")
    p.add_argument("--dry-run", action="store_true", help="create: только показать, что будет сделано")
    args = p.parse_args()

    if args.command != "check" and not (args.org and args.prefix):
        p.error("нужны --org и --prefix")
    if args.command == "collect" and not args.into:
        p.error("collect: нужен --into")

    students, bad = read_students(args.students)
    for b in bad:
        print("  ПЛОХОЙ ЛОГИН", b)
    print(f"Студентов с логином: {len(students)}\n")

    gh = GitHub(os.environ.get("GITHUB_TOKEN"), dry_run=args.dry_run)
    code = {"check": cmd_check, "create": cmd_create,
            "status": cmd_status, "collect": cmd_collect}[args.command](gh, students, args)
    # плохой логин в таблице — повод исправить её перед созданием репо, но не валить сбор работ
    return code or (1 if bad and args.command in ("check", "create") else 0)


if __name__ == "__main__":
    raise SystemExit(main())
