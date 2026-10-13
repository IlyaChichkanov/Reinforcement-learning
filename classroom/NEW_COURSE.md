# Та же схема сдачи для другого курса

Как завести для другого курса то же, что работает в курсе RL: у каждого студента закрытый репозиторий на весь курс, домашки приходят в него папками `hw1/`, `hw2/`, …, автопроверка ставит ✅/❌, работы сами собираются в закрытый «пульт» преподавателя, личные коды для квизов лежат у каждого в репозитории. Как это устроено — в [`SETUP.md`](SETUP.md). Ниже — что сделать для нового курса; пример везде — курс «Вычислительная оптимизация» ([`Numerical-Optimization`](https://github.com/IlyaChichkanov/Numerical-Optimization)).

## Что общее, а что своё у курса

**Общее — не копируется.** Скрипты (`tools/student_repos.py`, `check_homework.py`, `check_quiz_code.py`, `make_roster.py`) лежат в этом репозитории, и кнопки всех курсов берут их отсюда. Исправление здесь сразу работает во всех курсах.

**Своё у каждого курса:**

| Что | RL | Вычислительная оптимизация |
|---|---|---|
| Организация на GitHub | `rl-course-misis` | новая, например `numopt-course-misis` |
| Шаблон репозитория студента | `rl-template` | `numopt-template` |
| Репозиторий студента | `rl-<логин>` | `numopt-<логин>` |
| Пульт | `course-admin` | `course-admin` (в своей организации) |
| Где лежат задания | `homeworks/hw1/homework.ipynb` | `homeworks/hw03.ipynb` (+ `hw03.md`) |
| Страница квизов | `ilyachichkanov.github.io/Reinforcement-learning/` | `ilyachichkanov.github.io/Numerical-Optimization/` |
| Таблица логинов | своя | своя |

Отдельная организация на курс — чтобы списки студентов, репозитории и пульты не смешивались.

## Шаги преподавателя (≈30 минут, один раз)

### 1. Организация
[github.com/account/organizations/new](https://github.com/account/organizations/new) → **Free** → имя (например `numopt-course-misis`) → Create. Затем в Settings организации:
* **Member privileges → Base permissions → No permission**;
* **Actions → General → Allow all actions and reusable workflows**;
* **Personal access tokens → Settings → Allow access via fine-grained personal access tokens** (иначе на шаге 4 организации не будет в списке).

Студентов в организацию не добавлять — их приглашает кнопка, каждого только в его репозиторий.

### 2. Доступ Claude к организации
[github.com/apps/claude/installations/new](https://github.com/apps/claude/installations/new) → выбрать **новую организацию** → All repositories → Install.

### 3. Два пустых закрытых репозитория
[github.com/organizations/&lt;организация&gt;/repositories/new](https://github.com/organizations/numopt-course-misis/repositories/new) → **Private**, без README:
* `numopt-template` — шаблон (имя: `<префикс>-template`);
* `course-admin` — пульт.

### 4. Токен и секрет
[github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new):
* **Resource owner — новая организация**, не ваш личный аккаунт (в курсе RL это была ошибка, из-за которой кнопка не видела шаблон);
* Expiration — до конца семестра; Repository access — **All repositories**;
* Permissions → Repositories: **Administration** — Read and write, **Contents** — Read and write.

Скопировать токен → `course-admin` → Settings → Secrets and variables → Actions → **New repository secret**: имя `ADMIN_TOKEN`.

### 5. Переменные пульта
Там же, вкладка **Variables** → New repository variable:

| Переменная | Значение для «Вычислительной оптимизации» | Зачем |
|---|---|---|
| `REPO_PREFIX` | `numopt` | репозитории `numopt-<логин>`, шаблон `numopt-template` |
| `COURSE_REPO` | `IlyaChichkanov/Numerical-Optimization` | откуда брать задания |
| `HW_PATH` | `homeworks/hw{NN}.ipynb` | путь к заданию блока: `{N}` — номер, `{NN}` — с ведущим нулём |
| `QUIZ_GROUP` | `NUMOPT-2026` | группа для личных кодов; не менять посреди курса |
| `QUIZ_URL` | `https://ilyachichkanov.github.io/Numerical-Optimization/` | ссылка в файле с личным кодом |
| `CURRENT_BLOCK` | номер текущей домашки | какой блок собирается каждый час |

Без переменных кнопки работают как в курсе RL — поэтому в `rl-course-misis` их задавать не нужно.

### 6. Логины студентов
Попросите Claude сделать таблицу логинов для группы (как для RL) и разошлите студентам ссылку.

### 7. Дальше — Claude
Напишите в сессии Claude Code «подключи `<организация>/numopt-template` и `<организация>/course-admin`». Claude:
* заполнит `numopt-template`: автопроверка из [`template/`](template/) и README для студента со ссылками этого курса;
* заполнит `course-admin`: кнопки из [`admin/`](admin/), `students.csv` из таблицы, `roster.csv` с кодами (`make_roster.py --from-fio --group NUMOPT-2026`);
* сделает квизы (шаг 8) и обновит инструкцию для студентов в репозитории курса.

Вам останется: у `numopt-template` — Settings → **Template repository**.

### 8. Квизы
Страница квизов — своя у каждого курса, в его публичном репозитории:
* Claude пишет вопросы по лекциям (`quizzes/weekNN.json`, по 12 вопросов) и копирует `tools/build_quiz_page.py`, `quizcode.py`, `grade_quiz.py`; в копии меняются только название курса и ссылка на репозиторий. **Соль `SALT` в `quizcode.py` и на странице оставить прежней** — по ней автопроверка в репозиториях студентов сверяет коды;
* страница собирается в `docs/index.html`;
* вы включаете **GitHub Pages**: репозиторий курса → Settings → Pages → Deploy from a branch → `main` / `/docs`.

### 9. Запуск
В `course-admin` → Actions:
1. **«Создать репозитории студентов»** → Run workflow (флажок пустой) — репозитории и приглашения;
2. **«Выдать блок»** → номер домашки и **недели квизов** через запятую.

Про недели: по умолчанию кнопка кладёт заготовки кодов для недель 2N−1 и 2N, как в RL (домашка на две лекции). Если в курсе иначе — укажите явно. В «Вычислительной оптимизации» ДЗ 3 — по лекциям 4 и 5, то есть `4,5`; ДЗ 4 — `6,7`.

Отчёт каждой кнопки сохраняется в `course-admin/reports/` — Claude читает его сам, достаточно написать «проверь».

## Домашки с PDF

Ноутбук остаётся главным файлом: `hwN/homework.ipynb`, его проверяет автопроверка. Теорию «на бумаге» студент кладёт в ту же папку `hwN/` — фото или PDF с любым именем (например `hwN/theory.pdf`). Автопроверка её не смотрит, а сбор работ забирает **всю** папку `hwN/` — проверяющий получит и ноутбук, и PDF.

Если у блока есть описание рядом с ноутбуком (`homeworks/hw03.md` при `HW_PATH=homeworks/hw{NN}.ipynb`, или `README.md` в папке блока), студент получит его как `hwN/README.md`.

## Переход с другой схемы посреди семестра

В «Вычислительной оптимизации» сейчас сдают pull request'ом в публичный репозиторий (`submissions/hwNN/<фамилия>.ipynb`). Удобнее всего переключиться на границе задания: текущее (ДЗ 3, до 21.10) досдаётся как было, а начиная с ДЗ 4 — по новой схеме. Тогда:
* в `README.md` и `CONTRIBUTING.md` курса — новая инструкция «как сдавать» (Claude напишет по образцу [`../homeworks/SUBMIT.md`](../homeworks/SUBMIT.md));
* папку `submissions/` закрыть для новых работ, старые оставить.

Плюс новой схемы именно для этого курса: работы студентов перестают быть видны всем (в публичном репозитории их видит любой).

## Каждую неделю

То же, что в RL ([`SETUP.md`](SETUP.md), раздел «На каждый блок»): положить задание в репозиторий курса → `CURRENT_BLOCK` → «Выдать блок» → дедлайн в README курса → после дедлайна «Собрать работы» с `before` → «проверь блок N» Claude.
