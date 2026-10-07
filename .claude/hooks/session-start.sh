#!/bin/bash
# Подтягивает приватный репозиторий с инструкциями курса (rl_misis_private)
# и выводит главное в контекст сессии. Содержимое приватного репо в этот
# (публичный) репозиторий не коммитится: клон лежит рядом, скилл копируется
# в .claude/skills/course-materials, который в .gitignore.
set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PRIVATE_URL="https://github.com/IlyaChichkanov/rl_misis_private"
PRIVATE_DIR="${RL_PRIVATE_DIR:-$(dirname "$PROJECT_DIR")/rl_misis_private}"

if git -C "$PRIVATE_DIR" rev-parse HEAD >/dev/null 2>&1; then
  timeout 60 git -C "$PRIVATE_DIR" pull --ff-only --quiet >/dev/null 2>&1 \
    || echo "(не удалось обновить $PRIVATE_DIR, используется локальная копия)"
else
  timeout 120 git clone --quiet --depth 1 "$PRIVATE_URL" "$PRIVATE_DIR" >/dev/null 2>&1 \
    || rm -rf "$PRIVATE_DIR"
fi

if [ ! -f "$PRIVATE_DIR/CLAUDE.md" ]; then
  cat <<MSG
# Контекст курса не подтянут
Приватный репозиторий с инструкциями курса ($PRIVATE_URL) не удалось склонировать
(скорее всего, у сессии нет к нему доступа). До любой работы с материалами курса:
подключи репозиторий IlyaChichkanov/rl_misis_private (инструмент add_repo в облачной
сессии), склонируй его в $PRIVATE_DIR и прочитай CLAUDE.md, CONTEXT.md и
.claude/skills/course-materials/SKILL.md.
MSG
  exit 0
fi

# Скилл — в публичный репо (gitignored), чтобы он срабатывал как обычный скилл.
if [ -d "$PRIVATE_DIR/.claude/skills/course-materials" ]; then
  mkdir -p "$PROJECT_DIR/.claude/skills"
  rm -rf "$PROJECT_DIR/.claude/skills/course-materials"
  cp -r "$PRIVATE_DIR/.claude/skills/course-materials" "$PROJECT_DIR/.claude/skills/"
fi

cat <<MSG
# Контекст курса из приватного репозитория
Клон: $PRIVATE_DIR (коммит $(git -C "$PRIVATE_DIR" log -1 --format='%h %cs' 2>/dev/null)).
Обязательно прочитай целиком, прежде чем работать с лекциями, семинарами, домашками или квизами:
- $PRIVATE_DIR/CONTEXT.md — что уже есть в этом репо, программа по неделям, соглашения;
- $PRIVATE_DIR/.claude/skills/course-materials/SKILL.md — как делать лекцию и задания.
Правки инструкций и контекста коммить в приватный репозиторий, а не сюда.

---
MSG
cat "$PRIVATE_DIR/CLAUDE.md"
exit 0
