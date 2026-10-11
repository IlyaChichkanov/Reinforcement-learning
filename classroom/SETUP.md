# Сдача работ: схема для преподавателя

GitHub Classroom закрыт (сайт и API отключены 28 августа 2026), его заменяет эта схема на обычных средствах GitHub. Инструкция для студентов — [`../homeworks/SUBMIT.md`](../homeworks/SUBMIT.md).

## Как устроено

Всё живёт в организации курса **`rl-course-misis`**:

| Репозиторий | Кто видит | Что в нём |
|---|---|---|
| `rl-hw-block1` (шаблон) | преподаватель | `homework.ipynb`, `quiz/week01.txt`, `quiz/week02.txt`, автопроверка — содержимое [`template/`](template/) плюс ноутбук блока |
| `rl-hw-block1-<логин>` | студент и преподаватель | копия шаблона; сюда студент загружает работу |
| `course-admin` | преподаватель | «пульт»: список студентов, кнопки, собранные работы, оценки — содержимое [`admin/`](admin/) |

1. В `course-admin` преподаватель нажимает **«Создать репозитории студентов»**: для каждого логина из `students.csv` создаётся закрытый `rl-hw-block1-<логин>`, студенту приходит приглашение.
2. Студент принимает приглашение, загружает решённый ноутбук и коды квизов. Автопроверка в его репозитории ставит ✅ или ❌ (сдано полностью или нет).
3. Раз в час `course-admin` сам забирает работы в `block1/<логин>/`. Токены есть только в `course-admin`, в репозиториях студентов их нет.
4. Проверка: подключить `course-admin` к сессии Claude Code и попросить «проверь блок 1». Оценки и комментарии появятся в `block1/grades.csv` и `block1/<логин>/feedback.md`.

Логика — в [`../tools/student_repos.py`](../tools/student_repos.py) (`check`, `create`, `status`, `collect`), workflow в `course-admin` вызывают его.

## Один раз

### 1. Организация
Создана: `rl-course-misis`. В Settings организации:
* **Member privileges → Base permissions → No permission**;
* **Actions → General → Allow all actions and reusable workflows**.

Студентов в организацию **не** добавлять: они получают доступ только к своему репозиторию.

### 2. Два пустых закрытых репозитория
В организации: **New repository** → имя → **Private** → без README → **Create**. Нужны два:
* `rl-hw-block1`
* `course-admin`

Затем заполнить их: содержимое `template/` и `homeworks/hw1/homework.ipynb` — в `rl-hw-block1`, содержимое `admin/` и `students.csv` — в `course-admin`. Это может сделать Claude, если подключить оба репозитория к сессии.

У `rl-hw-block1`: **Settings → General → поставить галочку Template repository**.

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

## На каждый блок

1. Шаблон `rl-hw-blockN`: копия `template/` с ноутбуком блока; файлы квизов переименовать под недели блока (`week03.txt`, `week04.txt`, ...). Галочка Template repository.
2. В `course-admin`: переменная `CURRENT_BLOCK` = N; **Actions → Создать репозитории студентов → Run workflow** с номером блока. Первый раз удобно запустить с галочкой *dry_run*.
3. Дедлайн вписать в таблицу в [`../homeworks/README.md`](../homeworks/README.md).
4. После дедлайна: **Actions → Собрать работы → Run workflow** с полем `before` = момент дедлайна. Версии на этот момент лягут в `blockN-deadline/`.

## Сбор баллов за квизы

```bash
uv run python tools/grade_quiz.py course-admin/block1-deadline/ --roster roster.csv --out journal.csv
```

Скрипт печатает баллы по каждому студенту, средний балл и три самых трудных вопроса недели.

## Что автопроверка не делает

Она не считает баллы за квиз (правильных ответов в репозитории студента нет) и не проверяет правильность решения — только что ноутбук выполнен целиком, в нём не осталось заготовок и ячеек с ошибками, а коды квизов вставлены и разбираются.

Правильные ответы квизов лежат в `quizzes/week*.json` этого репозитория, то есть открыты всем. Это осознанный размен: без своего сервера тренировочный режим иначе невозможен. Квиз — проверка понимания и повод к разбору на семинаре, а не защищённый экзамен.
