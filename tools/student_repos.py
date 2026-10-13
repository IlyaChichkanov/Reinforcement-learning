"""Личные репозитории студентов в организации курса: создать, выдать блок, собрать работы.

Замена GitHub Classroom (закрыт 28.08.2026). У каждого студента один закрытый
репозиторий на весь курс — ``<org>/<prefix>-<логин>`` из шаблона ``<org>/<prefix>-template``
(для курса RL — ``rl-course-misis/rl-<логин>`` из ``rl-template``).
Домашки выдаются в него папками ``hw1/``, ``hw2/``, …: ноутбук и заготовки кодов квизов.

Список студентов — CSV, выгруженный из таблицы логинов: нужны столбцы с «ФИО» и
«логин» в названии (как «№,ФИО,GitHub-логин»); строки над заголовком и пустые
логины пропускаются.

    python tools/student_repos.py check   --students students.csv
    python tools/student_repos.py create  --students students.csv --org rl-course-misis
    python tools/student_repos.py issue   --students students.csv --org rl-course-misis --block 1 \\
                                          --source homeworks/hw1
    python tools/student_repos.py status  --students students.csv --org rl-course-misis --block 1
    python tools/student_repos.py collect --students students.csv --org rl-course-misis --block 1 \\
                                          --into block1 [--before 2026-10-28T23:59:59+03:00]

С ``--group`` (``create`` и ``issue``) в репозиторий студента кладётся ещё файл
``МОЙ_КОД_ДЛЯ_КВИЗОВ.txt`` с личным кодом для зачётного режима квизов. Код тот же, что
выдаёт ``make_roster.py --from-fio --group …`` по этому же students.csv.

Токен — из GITHUB_TOKEN: fine-grained, на организацию, права Administration и Contents
(запись), Metadata (чтение). Адрес API можно подменить через GITHUB_API_URL (для тестов).
``create`` и ``issue`` можно запускать сколько угодно раз: существующие репозитории,
приглашения и файлы пропускаются — работа студента никогда не перезаписывается.
"""

from __future__ import annotations

import argparse
import base64
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
NOTEBOOK = "homework.ipynb"
QUIZ_DIR = "quiz"
ISSUE_MESSAGE = "Выдан блок {block}"        # autograde.yml пропускает коммиты с таким началом

CODE_FILE = "МОЙ_КОД_ДЛЯ_КВИЗОВ.txt"
QUIZ_URL = "https://ilyachichkanov.github.io/Reinforcement-learning/"
CODE_TEXT = """\
Ваш личный код для квизов: {code}

Вводите его на странице квизов в режиме «Зачёт»:
{url}

Код личный — не передавайте его другим: по нему засчитываются ваши результаты.
"""

QUIZ_PLACEHOLDER = """\
# Код результата квиза по лекции {week}.
# Пройдите квиз в зачётном режиме, нажмите «Скопировать» и вставьте код
# одной строкой вместо этой подсказки. Строки, начинающиеся с #, игнорируются.
#
# Выглядеть должно так:  W{week}:IVAN23:BDAACBDABCAD:7F3A

вставьте сюда код результата
"""


@dataclass
class Student:
    name: str
    login: str      # нормализованный
    raw: str        # как вписан в таблицу — для сообщений

    def repo(self, org: str, prefix: str) -> str:
        return f"{org}/{prefix}-{self.login.lower()}"


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
    width = max(i_login, i_name) + 1
    for r in rows[header_at + 1:]:
        r = r + [""] * (width - len(r))
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


def quiz_codes(path: Path, group: str) -> dict[str, str]:
    """{«фамилия имя» в нижнем регистре: личный код} — как в make_roster.py --from-fio."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_roster import build_roster, read_fio
    return {f"{r['фамилия']} {r['имя']}".lower(): r["код"] for r in build_roster(read_fio(path, group))}


def code_file(s: Student, codes: dict[str, str], url: str = QUIZ_URL) -> dict[str, bytes]:
    key = " ".join(s.name.split()[:2]).lower()
    return {CODE_FILE: CODE_TEXT.format(code=codes[key], url=url).encode()} if key in codes else {}


def put_missing(gh: "GitHub", repo: str, files: dict[str, bytes], message: str) -> tuple[list[str], list[str], list[str]]:
    """Положить файлы, которых ещё нет. (добавлено, уже было, ошибки)."""
    put, kept, errs = [], [], []
    for path, content in files.items():
        status, _ = gh.get(f"/repos/{repo}/contents/{q(path)}")
        if status == 200:
            kept.append(path)          # уже есть — не трогаем, там может быть работа студента
            continue
        # в только что созданном из шаблона репо ветка появляется не сразу: 404/409 — подождать и повторить
        for attempt in range(5):
            status, data = gh.call("PUT", f"/repos/{repo}/contents/{q(path)}", {
                "message": message, "content": base64.b64encode(content).decode()})
            if status not in (404, 409):
                break
            time.sleep(2 + 2 * attempt)
        if status in (200, 201):
            put.append(path)
        else:
            errs.append(fail(f"{repo}/{path}", status, data))
    return put, kept, errs


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
                    raw, status = resp.read(), resp.status
                break
            except urllib.error.HTTPError as e:
                raw, status = e.read(), e.code
                retry = status in (502, 503, 504) or (
                    status in (403, 429) and e.headers.get("Retry-After"))
                if retry and attempt < 3:
                    time.sleep(int(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                break
        if accept.endswith("raw+json"):
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


def q(path: str) -> str:
    return urllib.parse.quote(path)


# ---------- команды ----------

def cmd_check(gh: GitHub, students: list[Student], args) -> int:
    missing = 0
    for s in students:
        status, data = gh.get(f"/users/{q(s.login)}")
        if status == 200:
            note = "" if data.get("login") == s.login else f" (на GitHub: {data.get('login')})"
            print(f"  ok        {s.login:24} {s.name}{note}")
        else:
            missing += 1
            what = "НЕ НАЙДЕН" if status == 404 else f"ошибка {status}"
            print(f"  {what:9} {s.raw:24} {s.name}")
    print(f"\nНайдено {len(students) - missing} из {len(students)}.")
    return 1 if missing else 0


def cmd_create(gh: GitHub, students: list[Student], args) -> int:
    template = args.template or f"{args.org}/{args.prefix}-template"
    if not gh.dry_run:
        status, data = gh.get(f"/repos/{template}")
        if status != 200:
            sys.exit(fail(f"шаблон {template} недоступен", status, data))
        if not data.get("is_template"):
            sys.exit(f"{template} не отмечен как Template repository (Settings → General)")

    codes = quiz_codes(args.students, args.group) if args.group else None
    created = existing = invited = errors = 0
    for s in students:
        repo = s.repo(args.org, args.prefix)
        if gh.dry_run:
            print(f"  [dry-run] {repo} ← {template}; пригласить {s.login} ({s.name})")
            continue

        status, data = gh.get(f"/users/{q(s.login)}")
        if status != 200:
            errors += 1
            print(f"  ПРОПУЩЕН  {s.name}: логин «{s.raw}» не найден на GitHub")
            continue

        status, data = gh.get(f"/repos/{repo}")
        if status == 404:
            status, data = gh.call("POST", f"/repos/{template}/generate", {
                "owner": args.org, "name": repo.split("/", 1)[1], "private": True,
                "description": f"Домашние задания: {s.name}"})
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
            status, data = gh.call("PUT", f"/repos/{repo}/collaborators/{q(s.login)}",
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
        if codes is not None:
            extra = code_file(s, codes, args.quiz_url)
            if not extra:
                print(f"            нет личного кода для «{s.name}»")
            _, _, errs = put_missing(gh, repo, extra, "Личный код для квизов")
            for e in errs:
                errors += 1
                print("  ОШИБКА    " + e)

    if gh.dry_run:
        print("\nПробный запуск — ничего не создано. Запустите без dry_run.")
    else:
        print(f"\nСоздано {created}, уже было {existing}, приглашений {invited}, ошибок {errors}.")
    return 1 if errors else 0


def block_files(args) -> dict[str, bytes]:
    """Что выдать в hwN/: ноутбук, описание (если есть) и заготовки кодов квизов.

    --source — каталог с homework.ipynb (и, если есть, README.md), как homeworks/hw1 курса RL,
    или сам ноутбук, как homeworks/hw03.ipynb; тогда описание — одноимённый .md рядом.
    У студента ноутбук всегда называется hwN/homework.ipynb, описание — hwN/README.md.
    """
    src = Path(args.source)
    nb, md = (src / NOTEBOOK, src / "README.md") if src.is_dir() else (src, src.with_suffix(".md"))
    if not nb.is_file() or nb.suffix != ".ipynb":
        sys.exit(f"{nb}: нет ноутбука блока (проверьте номер блока и путь к заданиям HW_PATH)")
    weeks = ([int(w) for w in args.weeks.split(",") if w.strip()] if args.weeks
             else [2 * args.block - 1, 2 * args.block])
    files = {f"hw{args.block}/{NOTEBOOK}": nb.read_bytes()}
    if md.is_file():
        files[f"hw{args.block}/README.md"] = md.read_bytes()
    for w in weeks:
        files[f"hw{args.block}/{QUIZ_DIR}/week{w:02d}.txt"] = QUIZ_PLACEHOLDER.format(week=w).encode()
    return files


def cmd_issue(gh: GitHub, students: list[Student], args) -> int:
    files = block_files(args)
    codes = quiz_codes(args.students, args.group) if args.group else None
    message = ISSUE_MESSAGE.format(block=args.block)
    print("Файлы блока: " + ", ".join(files) + (f", {CODE_FILE}" if codes is not None else "") + "\n")
    added = errors = 0
    for s in students:
        repo = s.repo(args.org, args.prefix)
        if gh.dry_run:
            print(f"  [dry-run] {repo}: положить {len(files)} файла, если их там нет")
            continue
        status, data = gh.get(f"/repos/{repo}")
        if status != 200:
            errors += 1
            print(f"  НЕТ РЕПО  {repo} — сначала «Создать репозитории студентов»")
            continue
        mine = dict(files)
        if codes is not None:
            mine.update(code_file(s, codes, args.quiz_url))
            if CODE_FILE not in mine:
                print(f"  {s.login:24} нет личного кода для «{s.name}»")
        put, kept, errs = put_missing(gh, repo, mine, message)
        for e in errs:
            errors += 1
            print("  ОШИБКА    " + e)
        added += len(put)
        state = f"добавлено {len(put)}" + (f", уже было {len(kept)}" if kept else "")
        print(f"  {s.login:24} {state}")
    if gh.dry_run:
        print("\nПробный запуск — ничего не выдано. Запустите без dry_run.")
    else:
        print(f"\nДобавлено файлов: {added}, ошибок {errors}.")
    return 1 if errors else 0


def cmd_status(gh: GitHub, students: list[Student], args) -> int:
    hw = f"hw{args.block}"
    print(f"  {'логин':24} {'приглашение':12} {'коммитов в ' + hw:15} последний  ФИО")
    for s in students:
        repo = s.repo(args.org, args.prefix)
        status, _ = gh.get(f"/repos/{repo}")
        if status != 200:
            print(f"  {s.login:24} {'нет репо':12} {'':15} {'':10} {s.name}")
            continue
        _, inv = gh.get(f"/repos/{repo}/invitations")
        pending = any(((i.get("invitee") or {}).get("login") or "").lower() == s.login.lower()
                      for i in (inv if isinstance(inv, list) else []))
        _, commits = gh.get(f"/repos/{repo}/commits?path={q(hw)}&per_page=100")
        mine = [c for c in (commits if isinstance(commits, list) else [])
                if ((c.get("author") or {}).get("login") or "").lower() == s.login.lower()]
        last = mine[0]["commit"]["committer"]["date"][:10] if mine else ""
        print(f"  {s.login:24} {'не принято' if pending else 'принято':12} "
              f"{len(mine):<15} {last:10} {s.name}")
    return 0


def fetch(gh: GitHub, repo: str, path: str, ref: str) -> bytes | None:
    status, data = gh.get(f"/repos/{repo}/contents/{q(path)}?ref={q(ref)}",
                          accept="application/vnd.github.raw+json")
    return data if status == 200 else None


def write_if_changed(path: Path, content: bytes) -> bool:
    if path.exists() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def cmd_collect(gh: GitHub, students: list[Student], args) -> int:
    hw = f"hw{args.block}"
    into = Path(args.into)
    changed = 0
    for s in students:
        repo = s.repo(args.org, args.prefix)
        until = f"&until={q(args.before)}" if args.before else ""
        status, data = gh.get(f"/repos/{repo}/commits?per_page=1{until}")
        if status != 200 or not data:
            why = f"нет коммитов до {args.before}" if args.before and status == 200 else "нет репозитория"
            print(f"  {s.login:24} {why}")
            continue
        commit = data[0]
        ref = commit["sha"]

        # вся папка блока: ноутбук, коды квизов и всё, что студент положил сам (например, PDF с теорией)
        status, tree = gh.get(f"/repos/{repo}/git/trees/{q(ref)}?recursive=1")
        paths = [t["path"] for t in (tree.get("tree", []) if status == 200 and isinstance(tree, dict) else [])
                 if t.get("type") == "blob" and t["path"].startswith(hw + "/")]
        if f"{hw}/{NOTEBOOK}" not in paths:
            print(f"  {s.login:24} блок {args.block} не выдан")
            continue
        files: dict[str, bytes] = {}
        for path in paths:
            content = fetch(gh, repo, path, ref)
            if content is not None:
                files[path[len(hw) + 1:]] = content

        # последний коммит студента в папке блока — когда он реально сдавал
        _, mine = gh.get(f"/repos/{repo}/commits?path={q(hw)}&sha={q(ref)}&per_page=100")
        mine = [c for c in (mine if isinstance(mine, list) else [])
                if ((c.get("author") or {}).get("login") or "").lower() == s.login.lower()]
        meta = {"login": s.login, "name": s.name, "repo": repo, "block": args.block,
                "snapshot_sha": ref,
                "submitted_at": mine[0]["commit"]["committer"]["date"] if mine else None,
                "student_commits": len(mine)}
        files["meta.json"] = (json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode()

        dest = into / s.login.lower()
        n = sum(write_if_changed(dest / rel, content) for rel, content in files.items())
        changed += n
        when = (meta["submitted_at"] or "не сдавал")[:16]
        print(f"  {s.login:24} {when:16} {'обновлено' if n else 'без изменений'}")
    print(f"\nИзменено файлов: {changed}.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Личные репозитории студентов в организации курса")
    p.add_argument("command", choices=["check", "create", "issue", "status", "collect"])
    p.add_argument("--students", type=Path, required=True, help="CSV с ФИО и GitHub-логинами")
    p.add_argument("--org", help="организация, например rl-course-misis")
    p.add_argument("--prefix", default="rl", help="репозитории <prefix>-<логин> (по умолчанию rl)")
    p.add_argument("--template", help="create: шаблон, по умолчанию <org>/<prefix>-template")
    p.add_argument("--block", type=int, help="issue/status/collect: номер блока")
    p.add_argument("--source", help="issue: каталог блока с homework.ipynb (homeworks/hw1) или сам ноутбук (homeworks/hw03.ipynb)")
    p.add_argument("--group", help="create/issue: группа для личного кода квизов (положить МОЙ_КОД_ДЛЯ_КВИЗОВ.txt)")
    p.add_argument("--quiz-url", default=QUIZ_URL, help="create/issue: страница квизов для файла с личным кодом")
    p.add_argument("--weeks", help="issue: недели квизов через запятую, по умолчанию 2N-1,2N")
    p.add_argument("--into", help="collect: куда складывать работы")
    p.add_argument("--before", help="collect: взять последний коммит не позже этого момента (ISO 8601)")
    p.add_argument("--dry-run", action="store_true", help="create/issue: только показать, что будет сделано")
    args = p.parse_args()

    if args.command != "check" and not args.org:
        p.error("нужен --org")
    if args.command in ("issue", "status", "collect") and not args.block:
        p.error(f"{args.command}: нужен --block")
    if args.command == "issue" and not args.source:
        p.error("issue: нужен --source")
    if args.command == "collect" and not args.into:
        p.error("collect: нужен --into")

    students, bad = read_students(args.students)
    for b in bad:
        print("  ПЛОХОЙ ЛОГИН", b)
    print(f"Студентов с логином: {len(students)}\n")

    gh = GitHub(os.environ.get("GITHUB_TOKEN"), dry_run=args.dry_run)
    commands = {"check": cmd_check, "create": cmd_create, "issue": cmd_issue,
                "status": cmd_status, "collect": cmd_collect}
    code = commands[args.command](gh, students, args)
    # плохой логин в таблице — повод исправить её перед созданием репо, но не валить сбор работ
    return code or (1 if bad and args.command in ("check", "create") else 0)


if __name__ == "__main__":
    raise SystemExit(main())
