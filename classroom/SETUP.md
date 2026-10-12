# Сдача работ: схема для преподавателя

GitHub Classroom закрыт (сайт и API отключены 28 августа 2026), его заменяет эта схема на обычных средствах GitHub. Инструкция для студентов — [`../homeworks/SUBMIT.md`](../homeworks/SUBMIT.md).

## Как устроено

Всё живёт в организации курса **`rl-course-misis`**:

| Репозиторий | Кто видит | Что в нём |
|---|---|---|
| `rl-template` (шаблон) | преподаватель | README для студента и автопроверка — содержимое [`template/`](template/) |
| `rl-<логин>` | студент и преподаватель | один на весь курс: копия шаблона, домашки — папками `hw1/`, `hw2/`, … |
| `course-admin` | преподаватель | «пульт»: список студентов, кнопки, собранные работы, оценки — содержимое [`admin/`](admin/) |

1. В начале курса в `course-admin` нажимается **«Создать репозитории студентов»**: для каждого логина из `students.csv` создаётся закрытый `rl-<логин>`, студенту приходит приглашение — одно на весь курс.
2. На каждый блок — **«Выдать блок»**: в каждый репозиторий добавляется `hwN/` с ноутбуком из [`../homeworks/hwN/`](../homeworks/) и заготовками кодов квизов. Существующие файлы не перезаписываются.
3. Студент загружает решённый ноутбук и коды квизов в `hwN/`. Автопроверка в его репозитории проверяет изменённые блоки и ставит ✅ или ❌ (сдано полностью или нет).
4. Раз в час `course-admin` сам забирает работы текущего блока в `blockN/<логин>/`. Токен есть только в `course-admin`, в репозиториях студентов его нет.
5. Проверка: подключить `course-admin` к сессии Claude Code и попросить «проверь блок 1». Оценки и комментарии появятся в `block1/grades.csv` и `block1/<логин>/feedback.md`.

Логика — в [`../tools/student_repos.py`](../tools/student_repos.py) (`check`, `create`, `issue`, `status`, `collect`), workflow в `course-admin` вызывают его.

## Один раз

### 1. Организация
Создана: `rl-course-misis`. В Settings организации:
* **Member privileges → Base permissions → No permission**;
* **Actions → General → Allow all actions and reusable workflows**.

Студентов в организацию **не** добавлять: они получают доступ только к своему репозиторию.

### 2. Два пустых закрытых репозитория
В организации: **New repository** → имя → **Private** → без README → **Create**. Нужны два:
* `rl-template`
* `course-admin`

Затем заполнить их: содержимое `template/` — в `rl-template`, содержимое `admin/` и `students.csv` — в `course-admin`. Это может сделать Claude, если подключить оба репозитория к сессии.

У `rl-template`: **Settings → General → поставить галочку Template repository**.

### 3. Токен
github.com → аватар → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**:
* **Token name**: `rl-course-admin`;
* **Expiration**: до конца семестра (например, 31.01.2027);
* **Resource owner**: `rl-course-misis` (не ваш личный аккаунт!);
* **Repository access**: *All repositories*;
* **Repository permissions**: *Administration* — Read and write; *Contents* — Read and write; *Metadata* — Read-only (ставится сам);
* **Generate token** → скопировать (показывается один раз).

Если организация требует одобрения токенов, одобрить его: Settings организации → *Personal access tokens → Pending requests*.

### 4. Секрет
`course-admin` → **Settings → Secrets and variables → Actions → New repository secret**: имя `ADMIN_TOKEN`, значение — токен.

### 5. Страница квизов
У `Reinforcement-learning`: **Settings → Pages → Deploy from a branch → `main` / `/docs`**.

### 6. Личные коды для квизов

```bash
uv run python tools/make_roster.py students.csv   # CSV со столбцами фамилия,имя,группа
```

`roster.csv` в публичный репозиторий не коммитится.

## Начало курса

В `course-admin`: **Actions → Создать репозитории студентов → Run workflow** (первый раз — с галочкой *dry_run*). Когда в таблице появляются новые логины — обновить `students.csv` и запустить ещё раз.

## На каждый блок

1. Ноутбук блока — в `homeworks/hwN/homework.ipynb` этого репозитория (в `main`).
2. В `course-admin`: **Actions → Выдать блок → Run workflow** с номером N. Переменная `CURRENT_BLOCK` = N, чтобы ежечасный сбор переключился на новый блок.
3. Дедлайн вписать в таблицу в [`../homeworks/README.md`](../homeworks/README.md).
4. После дедлайна: **Actions → Собрать работы → Run workflow** с номером блока и полем `before` = момент дедлайна. Версии на этот момент лягут в `blockN-deadline/`.

## Сбор баллов за квизы

```bash
uv run python tools/grade_quiz.py course-admin/block1-deadline/ --roster roster.csv --out journal.csv
```

Скрипт печатает баллы по каждому студенту, средний балл и три самых трудных вопроса недели.

## Что автопроверка не делает

Она проверяет только блоки, изменённые в пуше, не считает баллы за квиз (правильных ответов в репозитории студента нет) и не проверяет правильность решения — только что ноутбук выполнен целиком, в нём не осталось заготовок и ячеек с ошибками, а коды квизов вставлены и разбираются.

Правильные ответы квизов лежат в `quizzes/week*.json` этого репозитория, то есть открыты всем. Это осознанный размен: без своего сервера тренировочный режим иначе невозможен. Квиз — проверка понимания и повод к разбору на семинаре, а не защищённый экзамен.
