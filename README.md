# Reinforcement Learning

Курс по обучению с подкреплением для магистров. Полная программа — в [Syllabus.md](Syllabus.md).

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
<td>Введение в RL, многорукие бандиты, MDP</td>
<td><a href="01-intro-mdp-bandits/lecture/lecture.ipynb">лекция</a> · <a href="01-intro-mdp-bandits/seminar/seminar.ipynb">семинар</a> · <a href="01-intro-mdp-bandits/homework/homework.ipynb">дз</a></td>
</tr>
<tr>
<td>2</td>
<td>08.09</td>
<td>Динамическое программирование</td>
<td></td>
</tr>
<tr>
<td>3</td>
<td>15.09</td>
<td>Monte Carlo и TD-обучение</td>
<td></td>
</tr>
<tr>
<td>4</td>
<td>22.09</td>
<td>Аппроксимация функций, TD(λ)</td>
<td></td>
</tr>
<tr>
<td>5</td>
<td>29.09</td>
<td>Deep Q-Learning</td>
<td></td>
</tr>
<tr>
<td>6</td>
<td>06.10</td>
<td>Policy Gradient методы</td>
<td></td>
</tr>
<tr>
<td>7</td>
<td>13.10</td>
<td>TRPO и PPO</td>
<td></td>
</tr>
<tr>
<td>8</td>
<td>20.10</td>
<td>Непрерывное управление: DDPG, TD3, SAC</td>
<td></td>
</tr>
<tr>
<td>9</td>
<td>27.10</td>
<td>Model-Based RL</td>
<td></td>
</tr>
<tr>
<td>10</td>
<td>03.11</td>
<td>Exploration в глубоком RL</td>
<td></td>
</tr>
<tr>
<td>11</td>
<td>10.11</td>
<td>Imitation Learning и Offline RL</td>
<td></td>
</tr>
<tr>
<td>12</td>
<td>17.11</td>
<td>Multi-Agent RL</td>
<td></td>
</tr>
<tr>
<td>13</td>
<td>24.11</td>
<td>Distributional RL, Hierarchical/Meta-RL, POMDP</td>
<td></td>
</tr>
<tr>
<td>14</td>
<td>01.12</td>
<td>RL для LLM: RLHF и DPO</td>
<td></td>
</tr>
<tr>
<td>15</td>
<td>08.12</td>
<td>Фронтир RL, защита проектов</td>
<td></td>
</tr>
</tbody>
</table>

### Структура репозитория

Каждая неделя — отдельная папка `NN-topic-name/` с тремя частями:

* `lecture/` — конспект лекции в виде ноутбука (теория, вывод формул, иногда мини-демо)
* `seminar/` — ноутбук для занятия: разбор кода и эксперименты вживую
* `homework/` — ноутбук с заданием; часть проверок оформлена как `assert`, чтобы можно было проверить себя локально

### Окружение

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Основной стек: [Gymnasium](https://gymnasium.farama.org/) для сред, [PyTorch](https://pytorch.org/) для нейросетевых методов.
