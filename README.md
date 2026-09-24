# Reinforcement Learning

Курс по обучению с подкреплением для бакалавров 4 курса. Полная программа — в [Syllabus.md](Syllabus.md).

### Материалы занятий

<table>
<thead>
<tr>
<th>#</th>
<th>Дата</th>
<th>Тема</th>
<th>Материалы</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>01.09</td>
<td>Знакомство с Reinforcement Learning</td>
<td><a href="01-intro/lecture/lecture.ipynb">лекция</a> · <a href="01-intro/seminar/seminar.ipynb">семинар</a> · <a href="01-intro/seminar/dl_basics.ipynb">dl-минимум</a> · <a href="01-intro/seminar/pytorch_intro.ipynb">pytorch-разминка</a> · <a href="https://ilyachichkanov.github.io/Reinforcement-learning/">квиз</a></td>
</tr>
<tr>
<td>2</td>
<td>08.09</td>
<td>Ценность действия и уравнение Беллмана. Своя среда как MDP</td>
<td><a href="02-environments/lecture/lecture.ipynb">лекция</a> · <a href="02-environments/seminar/seminar.ipynb">семинар</a> · <a href="homeworks/hw1/homework.ipynb">дз блока 1</a> · <a href="https://ilyachichkanov.github.io/Reinforcement-learning/">квиз</a></td>
</tr>
<tr>
<td>3</td>
<td>15.09</td>
<td>Основные алгоритмы RL: value based</td>
<td></td>
</tr>
<tr>
<td>4</td>
<td>22.09</td>
<td>Основные алгоритмы RL: policy based</td>
<td></td>
</tr>
<tr>
<td>5</td>
<td>29.09</td>
<td>Введение в Deep Reinforcement Learning</td>
<td></td>
</tr>
<tr>
<td>6</td>
<td>06.10</td>
<td>Deep Q-Network (DQN)</td>
<td></td>
</tr>
<tr>
<td>7</td>
<td>13.10</td>
<td>Deep Policy Gradient (PG)</td>
<td></td>
</tr>
<tr>
<td>8</td>
<td>20.10</td>
<td>Actor-Critic</td>
<td></td>
</tr>
<tr>
<td>9</td>
<td>27.10</td>
<td>TRPO → PPO</td>
<td></td>
</tr>
<tr>
<td>10</td>
<td>03.11</td>
<td>DDPG → TD3 → LSTM-TD3</td>
<td></td>
</tr>
<tr>
<td>11</td>
<td>10.11</td>
<td>Model-based RL, часть 1</td>
<td></td>
</tr>
<tr>
<td>12</td>
<td>17.11</td>
<td>Model-based RL, часть 2</td>
<td></td>
</tr>
<tr>
<td>13</td>
<td>24.11</td>
<td>Иерархическое обучение с подкреплением</td>
<td></td>
</tr>
<tr>
<td>14</td>
<td>01.12</td>
<td>Выбор темы и организация проектной работы</td>
<td></td>
</tr>
<tr>
<td>15</td>
<td>08.12</td>
<td>Многоагентное обучение и кооперация агентов</td>
<td></td>
</tr>
<tr>
<td>16</td>
<td>15.12</td>
<td>Трансформеры в RL: decision transformers и action transformers. Защита проектов</td>
<td></td>
</tr>
</tbody>
</table>

### Структура репозитория

Каждая неделя — отдельная папка `NN-topic-name/` с двумя частями:

* `lecture/` — конспект лекции в виде ноутбука (теория, вывод формул, иногда мини-демо)
* `seminar/` — ноутбук для занятия: разбор кода и эксперименты вживую

Домашние задания лежат отдельно, в [`homeworks/`](homeworks/): одно задание на блок из двух лекций.

### Домашние задания

Домашка выдаётся **раз в две недели** и охватывает обе лекции блока: [`homeworks/hw1/homework.ipynb`](homeworks/hw1/homework.ipynb) — по лекциям 1–2 (Cross-Entropy, оценка политики, своя среда, уравнения Беллмана). Часть проверок оформлена как `assert`, чтобы можно было проверить себя локально. Как сдавать — в [`homeworks/README.md`](homeworks/README.md).

### Квизы

После каждой лекции — квиз на двенадцать вопросов: [страница квизов](https://ilyachichkanov.github.io/Reinforcement-learning/) (адрес заработает, когда в настройках репозитория включены GitHub Pages из папки `/docs` — см. [`classroom/SETUP.md`](classroom/SETUP.md)). В тренировочном режиме разбор появляется сразу, в зачётном страница выдаёт код результата, который сдаётся вместе с домашним заданием. Вопросы лежат в [`quizzes/`](quizzes/), схема сдачи — в [`classroom/SETUP.md`](classroom/SETUP.md).

### Если вы не работали с нейросетями

Нейросети появляются в курсе на неделе 5. Чтобы подойти к ней подготовленным, есть два ноутбука для самостоятельной работы, по 2–3 часа каждый:

* [`01-intro/seminar/dl_basics.ipynb`](01-intro/seminar/dl_basics.ipynb) — нейросети с нуля: модель с параметрами, функция потерь, градиентный спуск, backprop, первая сеть на PyTorch, классификация с кросс-энтропией. Внутри — дорожная карта на четыре недели с внешними ресурсами и упражнения с проверками (они же бонус В домашнего задания блока 1).
* [`01-intro/seminar/pytorch_intro.ipynb`](01-intro/seminar/pytorch_intro.ipynb) — механика PyTorch, которая нужна именно в RL: `gather`, `detach`, распределения, behavior cloning на CartPole.

### Окружение

Рекомендуемый способ — через [uv](https://docs.astral.sh/uv/):

```bash
cd RL
uv sync
uv run python -m ipykernel install --user --name rl-course --display-name "RL course"
```

Без uv — обычный venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Основной стек: [Gymnasium](https://gymnasium.farama.org/) для сред, [PyTorch](https://pytorch.org/) для нейросетевых методов.

**Google Colab.** Любой ноутбук можно открыть без установки: кнопка «Open in Colab» стоит в первой ячейке, а первая ячейка с кодом сама ставит недостающие пакеты. Ссылка для любого файла — `https://colab.research.google.com/github/IlyaChichkanov/Reinforcement-learning/blob/main/<путь к ноутбуку>`.
