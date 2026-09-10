"""Генерация иллюстраций для лекций курса.

Запуск из корня репозитория:

    python assets/make_figures.py

Все картинки рисуются в matplotlib без внешних зависимостей, поэтому их можно
перегенерировать после правок (например, если поменялась программа курса).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
DPI = 160

# Нейтральная палитра
C_AGENT = "#4C72B0"
C_ENV = "#55A868"
C_REWARD = "#DD8452"
C_TEXT = "#222222"
C_GREY = "#888888"
C_LIGHT = "#F2F2F2"
C_ACCENT = "#C44E52"
C_PURPLE = "#8172B2"
C_TEAL = "#64B5CD"


def _box(ax, xy, w, h, text, color, fontsize=13, text_color="white", weight="bold", radius=0.03):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=0, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight)


def _arrow(ax, p0, p1, color, text=None, text_offset=(0, 0), fontsize=12, rad=0.0, lw=2.5):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=22, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + text_offset[0], (p0[1] + p1[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=color, weight="bold")


def agent_env_loop():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.08, 0.35), 0.28, 0.3, "Агент\n(политика π)", C_AGENT, fontsize=15)
    _box(ax, (0.64, 0.35), 0.28, 0.3, "Среда\n(P, R)", C_ENV, fontsize=15)

    # действие: агент -> среда (верхняя дуга)
    _arrow(ax, (0.36, 0.58), (0.64, 0.58), C_AGENT, rad=-0.45, lw=3)
    ax.text(0.5, 0.86, "действие $a_t$", ha="center", fontsize=14, color=C_AGENT, weight="bold")

    # состояние и награда: среда -> агент (нижняя дуга)
    _arrow(ax, (0.64, 0.42), (0.36, 0.42), C_ENV, rad=-0.45, lw=3)
    ax.text(0.5, 0.12, "наблюдение $s_{t+1}$, награда $r_{t+1}$",
            ha="center", fontsize=14, color=C_ENV, weight="bold")

    ax.text(0.5, 0.5, "$t \\to t+1$", ha="center", va="center", fontsize=13, color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "agent_env_loop.png", dpi=DPI)
    plt.close(fig)


def ml_paradigms():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    titles = ["Supervised learning", "Unsupervised learning", "Reinforcement learning"]
    colors = [C_AGENT, C_PURPLE, C_REWARD]
    lines = [
        ["Дано: пары (x, y)", "Учитель говорит\nправильный ответ", "Цель: предсказывать y",
         "Данные фиксированы, i.i.d."],
        ["Дано: только x", "Учителя нет", "Цель: найти структуру\n(кластеры, плотность)",
         "Данные фиксированы, i.i.d."],
        ["Дано: среда, с которой\nможно взаимодействовать", "Обратная связь — награда:\nчисло, часто с задержкой",
         "Цель: максимизировать\nсуммарную награду", "Данные порождает сам агент"],
    ]
    for ax, title, color, body in zip(axes, titles, colors, lines):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        _box(ax, (0.03, 0.82), 0.94, 0.15, title, color, fontsize=14)
        y = 0.7
        for i, t in enumerate(body):
            ax.text(0.5, y, t, ha="center", va="top", fontsize=11.5,
                    color=C_TEXT if i < 3 else C_GREY, style="normal" if i < 3 else "italic")
            y -= 0.19
    fig.tight_layout()
    fig.savefig(OUT / "ml_paradigms.png", dpi=DPI)
    plt.close(fig)


def rl_timeline():
    events = [
        (1992, "TD-Gammon", "нейросеть + TD(λ) играет\nв нарды на уровне чемпионов"),
        (2013, "DQN (Atari)", "одна сеть учится играть\nв 49 игр по пикселям"),
        (2016, "AlphaGo", "победа над Ли Седолем;\nGo считался «нерешаемым»"),
        (2017, "AlphaZero", "self-play с нуля:\nшахматы, сёги, Go"),
        (2019, "OpenAI Five,\nAlphaStar", "Dota 2 и StarCraft II\nна уровне профессионалов"),
        (2022, "ChatGPT / RLHF", "RL на человеческих\nпредпочтениях для LLM"),
        (2024, "o1, DeepSeek-R1", "RL на проверяемых наградах:\nмодели учатся рассуждать"),
    ]
    n = len(events)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.plot([-0.5, n - 0.5], [0, 0], color=C_GREY, lw=2.5, zorder=1)
    for i, (year, name, desc) in enumerate(events):
        s_ = 1 if i % 2 == 0 else -1
        ax.plot([i, i], [0, 0.35 * s_], color=C_GREY, lw=1.5, zorder=1)
        ax.scatter([i], [0], s=110, color=C_ACCENT, zorder=3)
        ax.text(i, -0.17 * s_, str(year), ha="center", va="center", fontsize=11, color=C_TEXT)
        ax.text(i, 0.45 * s_, name, ha="center", va="bottom" if s_ > 0 else "top",
                fontsize=12.5, weight="bold", color=C_AGENT)
        ax.text(i, 0.95 * s_, desc, ha="center", va="bottom" if s_ > 0 else "top",
                fontsize=10, color=C_TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "rl_timeline.png", dpi=DPI)
    plt.close(fig)


def rl_taxonomy():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.38, 0.86), 0.24, 0.1, "Методы RL", C_TEXT, fontsize=15)

    # уровень 1
    _box(ax, (0.03, 0.62), 0.2, 0.1, "Табличные\n(недели 1-4)", C_GREY, fontsize=11)
    _box(ax, (0.27, 0.62), 0.28, 0.1, "Model-free deep RL", C_AGENT, fontsize=12)
    _box(ax, (0.59, 0.62), 0.18, 0.1, "Model-based\n(недели 11-12)", C_ENV, fontsize=11)
    _box(ax, (0.8, 0.62), 0.18, 0.1, "Трансформеры,\noffline (неделя 16)", C_PURPLE, fontsize=11)
    for x in (0.13, 0.41, 0.68, 0.89):
        _arrow(ax, (0.5, 0.86), (x, 0.72), C_GREY, lw=1.5)

    # табличные
    ax.text(0.13, 0.55, "бандиты, Cross-Entropy,\nDP, Monte Carlo, TD,\nSARSA, Q-learning", ha="center", va="top",
            fontsize=10, color=C_TEXT)
    # model-based
    ax.text(0.68, 0.55, "Dyna, MBPO,\nMuZero, Dreamer", ha="center", va="top", fontsize=10, color=C_TEXT)
    # offline
    ax.text(0.89, 0.55, "Decision Transformer,\nTrajectory Transformer,\nACT", ha="center", va="top", fontsize=10, color=C_TEXT)

    # уровень 2 под model-free
    _box(ax, (0.2, 0.36), 0.14, 0.09, "Value-based\n(нед. 5-6)", C_TEAL, fontsize=10, text_color=C_TEXT)
    _box(ax, (0.34, 0.36), 0.14, 0.09, "Policy-based\n(нед. 7, 9)", C_TEAL, fontsize=10, text_color=C_TEXT)
    _box(ax, (0.48, 0.36), 0.14, 0.09, "Actor-Critic\n(нед. 8-10)", C_TEAL, fontsize=10, text_color=C_TEXT)
    for x in (0.27, 0.41, 0.55):
        _arrow(ax, (0.41, 0.62), (x, 0.45), C_GREY, lw=1.5)
    ax.text(0.27, 0.32, "DQN, Double DQN,\nDueling, Rainbow", ha="center", va="top", fontsize=9.5, color=C_TEXT)
    ax.text(0.41, 0.32, "REINFORCE,\nTRPO, PPO", ha="center", va="top", fontsize=9.5, color=C_TEXT)
    ax.text(0.55, 0.32, "A2C, DDPG,\nTD3, SAC", ha="center", va="top", fontsize=9.5, color=C_TEXT)

    # нижняя полоса: расширения
    _box(ax, (0.03, 0.04), 0.95, 0.12,
         "Расширения: иерархический RL (нед. 13)  ·  проектная работа (нед. 14)  ·  multi-agent RL (нед. 15)  ·  "
         "трансформеры в RL (нед. 16)",
         C_LIGHT, fontsize=10.5, text_color=C_TEXT, weight="normal")

    fig.tight_layout()
    fig.savefig(OUT / "rl_taxonomy.png", dpi=DPI)
    plt.close(fig)


def mdp_gridworld():
    # 4x4 gridworld: старт, стены, яма (-1), выход (+1)
    n = 4
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    walls = {(1, 1), (2, 3)}
    goal = (3, 3)
    pit = (3, 1)
    start = (0, 0)
    for i in range(n):
        for j in range(n):
            color = "white"
            if (i, j) in walls:
                color = "#555555"
            elif (i, j) == goal:
                color = "#B5E0B5"
            elif (i, j) == pit:
                color = "#F4B6B6"
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=color, edgecolor=C_GREY, lw=1.5))
    ax.text(goal[0] + 0.5, goal[1] + 0.5, "выход\n+1", ha="center", va="center", fontsize=13, weight="bold", color=C_ENV)
    ax.text(pit[0] + 0.5, pit[1] + 0.5, "яма\n-1", ha="center", va="center", fontsize=13, weight="bold", color=C_ACCENT)
    ax.text(start[0] + 0.5, start[1] + 0.5, "старт", ha="center", va="center", fontsize=12, color=C_TEXT)
    ax.scatter([start[0] + 0.5], [start[1] + 0.25], s=380, color=C_AGENT, zorder=3)
    # действия из клетки (1,2)
    cx, cy = 1.5, 2.5
    for dx, dy, lab in ((0.42, 0, "→"), (-0.42, 0, "←"), (0, 0.42, "↑"), (0, -0.42, "↓")):
        _arrow(ax, (cx, cy), (cx + dx, cy + dy), C_AGENT, lw=2)
    ax.text(cx + 0.3, cy + 0.3, "a", ha="center", fontsize=12, color=C_AGENT, weight="bold")
    ax.text(2.0, -0.3,
            "Состояние = клетка, действие = одно из 4 направлений.\n"
            "Награда: −0.04 за каждый шаг (чтобы не гулять вечно),\n"
            "+1 за выход, −1 за яму. Переход «скользкий»: с p = 0.8\n"
            "идём куда хотели, с p = 0.1 — в каждую из боковых сторон.",
            ha="center", va="top", fontsize=9.5, color=C_TEXT)
    ax.set_ylim(-1.5, n)
    fig.tight_layout()
    fig.savefig(OUT / "mdp_gridworld.png", dpi=DPI)
    plt.close(fig)


def course_map():
    blocks = [
        ("Основы", C_GREY, ["1. Знакомство с RL", "2. Ключевые понятия,\n    построение среды",
                            "3. Алгоритмы RL:\n    value based", "4. Алгоритмы RL:\n    policy based"]),
        ("Deep RL", C_AGENT, ["5. Введение в Deep RL", "6. DQN", "7. Deep Policy Gradient", "8. Actor-Critic"]),
        ("Продвинутые методы", C_ENV, ["9. TRPO → PPO", "10. DDPG → TD3 → LSTM-TD3",
                                       "11. Model-based RL, ч. 1", "12. Model-based RL, ч. 2"]),
        ("Расширения и проект", C_REWARD, ["13. Иерархический RL", "14. Проектная работа:\n      выбор темы",
                                          "15. Multi-agent RL", "16. Трансформеры в RL,\n      защита проектов"]),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w = 0.225
    gap = 0.03
    for k, (title, color, items) in enumerate(blocks):
        x = 0.01 + k * (w + gap)
        _box(ax, (x, 0.85), w, 0.12, title, color, fontsize=13.5)
        ax.add_patch(FancyBboxPatch((x, 0.04), w, 0.75, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        y = 0.69
        for item in items:
            ax.text(x + 0.015, y, item, ha="left", va="center", fontsize=11.5, color=C_TEXT, linespacing=1.4)
            y -= 0.175
        if k < len(blocks) - 1:
            ax.text(x + w + gap / 2, 0.91, "→", ha="center", va="center", fontsize=18, color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "course_map.png", dpi=DPI)
    plt.close(fig)


def rl_origins():
    """Две ветки, из которых вырос RL: психология обучения и оптимальное управление."""
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # верхняя ветка: психология
    _box(ax, (0.01, 0.80), 0.2, 0.11, "Психология\nобучения", C_REWARD, fontsize=12.5)
    psych = [
        (0.24, "1890-1900-е\nПавлов", "условный рефлекс:\nстимул → реакция"),
        (0.43, "1898-1911\nТорндайк", "закон эффекта:\nдействие с приятным\nпоследствием повторяется"),
        (0.62, "1930-50-е\nСкиннер", "оперантное обусловливание,\nтермин «подкрепление»\n(reinforcement)"),
    ]
    for x, name, desc in psych:
        _box(ax, (x, 0.80), 0.16, 0.11, name, C_LIGHT, fontsize=10.5, text_color=C_TEXT)
        ax.text(x + 0.08, 0.77, desc, ha="center", va="top", fontsize=9, color=C_TEXT)
    for x0, x1 in ((0.21, 0.24), (0.40, 0.43), (0.59, 0.62)):
        _arrow(ax, (x0, 0.855), (x1, 0.855), C_GREY, lw=1.5)

    # нижняя ветка: оптимальное управление и ранний ИИ
    _box(ax, (0.01, 0.30), 0.2, 0.11, "Оптимальное\nуправление и ИИ", C_AGENT, fontsize=12.5)
    ctrl = [
        (0.24, "1950-е\nБеллман", "динамическое\nпрограммирование,\nмарковские процессы решений"),
        (0.43, "1954-61\nМинский", "обучающаяся машина\nSNARC; проблема\ncredit assignment"),
        (0.62, "1970-80-е\nКлопф, Саттон,\nБарто", "«гедонистические» нейроны,\nactor-critic,\nTD-обучение (1988)"),
    ]
    for x, name, desc in ctrl:
        _box(ax, (x, 0.30), 0.16, 0.11, name, C_LIGHT, fontsize=10.5, text_color=C_TEXT)
        ax.text(x + 0.08, 0.27, desc, ha="center", va="top", fontsize=9, color=C_TEXT)
    for x0, x1 in ((0.21, 0.24), (0.40, 0.43), (0.59, 0.62)):
        _arrow(ax, (x0, 0.355), (x1, 0.355), C_GREY, lw=1.5)

    # слияние
    _box(ax, (0.82, 0.50), 0.17, 0.16, "Reinforcement\nLearning\n1980-90-е", C_ENV, fontsize=12.5)
    ax.text(0.99, 0.47, "Q-learning (Уоткинс, 1989),\nTD-Gammon (Тезауро, 1992),\n"
                         "учебник Саттона и Барто (1998)", ha="right", va="top", fontsize=9, color=C_TEXT)
    _arrow(ax, (0.78, 0.855), (0.86, 0.67), C_REWARD, rad=-0.2, lw=2.2)
    _arrow(ax, (0.78, 0.355), (0.815, 0.53), C_AGENT, rad=-0.25, lw=2.2)

    ax.text(0.5, 0.03,
            "Слово «подкрепление» пришло из психологии; математика (ценность состояния, уравнения Беллмана) — "
            "из теории управления.\nRL как отдельная область появился, когда эти две линии соединились.",
            ha="center", va="bottom", fontsize=10, color=C_GREY, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "rl_origins.png", dpi=DPI)
    plt.close(fig)

def markov_chain():
    """Марковская цепь «день студента»: 4 состояния, вероятности переходов на стрелках."""
    import numpy as np
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {"Лекция": (0.2, 0.72), "Соцсети": (0.2, 0.22), "Сон": (0.62, 0.22), "Экзамен\nсдан": (0.62, 0.72)}
    colors = {"Лекция": C_AGENT, "Соцсети": C_ACCENT, "Сон": C_PURPLE, "Экзамен\nсдан": C_ENV}
    r = 0.09
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, facecolor=colors[name], edgecolor="white", lw=2, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=11, color="white", weight="bold", zorder=4)

    def edge(a, b, p, rad=0.0, off=(0, 0)):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        dx, dy = x1 - x0, y1 - y0
        d = (dx ** 2 + dy ** 2) ** 0.5
        p0 = (x0 + dx / d * r, y0 + dy / d * r)
        p1 = (x1 - dx / d * r, y1 - dy / d * r)
        arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, lw=2, color=C_GREY,
                              connectionstyle=f"arc3,rad={rad}", zorder=2)
        ax.add_patch(arr)
        mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
        ax.text(mx, my, p, ha="center", va="center", fontsize=11, color=C_TEXT, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), zorder=5)

    edge("Лекция", "Соцсети", "0.5", rad=0.3, off=(-0.05, 0))
    edge("Соцсети", "Лекция", "0.3", rad=0.3, off=(0.05, 0))
    edge("Лекция", "Экзамен\nсдан", "0.3", off=(0, 0.04))
    edge("Лекция", "Сон", "0.2", rad=0.15, off=(0.06, 0.03))
    edge("Соцсети", "Сон", "0.7", off=(0, -0.04))
    edge("Сон", "Лекция", "1.0", rad=0.15, off=(-0.06, -0.03))

    # петля у «Экзамен сдан» (терминальное состояние)
    ax.annotate("", xy=(0.66, 0.815), xytext=(0.58, 0.815),
                arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=2, connectionstyle="arc3,rad=-1.8"))
    ax.text(0.62, 0.93, "1.0", ha="center", va="center", fontsize=11, color=C_TEXT, weight="bold")

    ax.text(0.97, 0.5,
            "Марковская цепь:\nсостояния + вероятности\nпереходов между ними.\n\n"
            "Из каждого состояния\nсумма исходящих\nвероятностей равна 1.",
            ha="right", va="center", fontsize=10.5, color=C_TEXT, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "markov_chain.png", dpi=DPI)
    plt.close(fig)


def cem_loop():
    """Три шага метода Cross-Entropy: сыграть -> отобрать элиту -> подстроить политику."""
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.16)
    ax.axis("off")

    panels = [(0.03, "1. Сыграть", C_AGENT), (0.37, "2. Отобрать элиту", C_ENV),
              (0.71, "3. Подстроить политику", C_REWARD)]
    pw = 0.26
    for x0, title, color in panels:
        _box(ax, (x0, 0.83), pw, 0.11, title, color, fontsize=13)

    # Облако эпизодов: y — return. Большинство внизу, несколько удачных вверху.
    n = 34
    xs = rng.uniform(0.04, 0.24, n)
    ys = np.concatenate([rng.uniform(0.30, 0.45, n - 4), rng.uniform(0.62, 0.72, 4)])
    thr = 0.55

    # Панель 1: все эпизоды одинаковым цветом
    ax.scatter(xs + 0.005, ys, s=42, color=C_AGENT, alpha=0.85, zorder=3)
    ax.text(0.16, 0.20, "$K$ эпизодов\nтекущей политикой", ha="center", va="top",
            fontsize=10.5, color=C_TEXT, linespacing=1.5)
    ax.annotate("", xy=(0.028, 0.72), xytext=(0.028, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.6))
    ax.text(0.019, 0.51, "return", ha="center", va="center", fontsize=9.5,
            color=C_GREY, rotation=90)

    # Панель 2: те же точки, порог, элита выделена
    xs2 = xs + 0.34
    elite = ys >= thr
    ax.scatter(xs2[~elite], ys[~elite], s=38, color=C_GREY, alpha=0.35, zorder=3)
    ax.scatter(xs2[elite], ys[elite], s=95, color=C_ENV, alpha=0.95, zorder=4,
               edgecolors="white", linewidths=1.2)
    ax.plot([0.37, 0.63], [thr, thr], ls="--", lw=1.8, color=C_ACCENT, zorder=2)
    ax.text(0.632, thr, "порог", ha="left", va="center", fontsize=10,
            color=C_ACCENT, weight="bold")
    ax.text(0.50, 0.20, "квантиль уровня $q$:\nоставляем только лучшие", ha="center", va="top",
            fontsize=10.5, color=C_TEXT, linespacing=1.5)

    # Панель 3: строка политики до и после
    for x0, label, probs, color in [(0.74, "было", [0.25, 0.25, 0.25, 0.25], C_GREY),
                                    (0.87, "стало", [0.05, 0.70, 0.15, 0.10], C_REWARD)]:
        ax.text(x0 + 0.045, 0.70, label, ha="center", va="bottom", fontsize=10.5,
                color=C_TEXT, weight="bold")
        for k, pr in enumerate(probs):
            h = pr * 0.42
            ax.add_patch(Rectangle((x0 + k * 0.024, 0.26), 0.019, h,
                                   facecolor=color, edgecolor="none"))
        ax.text(x0 + 0.045, 0.235, "←  ↓  →  ↑", ha="center", va="top", fontsize=10,
                color=C_GREY)
    ax.text(0.86, 0.16, "частоты действий элиты\nстановятся новой политикой", ha="center",
            va="top", fontsize=10.5, color=C_TEXT, linespacing=1.5)

    # Возврат к шагу 1 — дугой поверх панелей, чтобы не задевать подписи
    _arrow(ax, (0.90, 0.96), (0.13, 0.96), C_GREY, rad=0.10, lw=2.0)
    ax.text(0.515, 1.11, "и снова, уже с новой политикой", ha="center", va="center",
            fontsize=11.5, color=C_GREY, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "cem_loop.png", dpi=DPI)
    plt.close(fig)


def env_anatomy():
    """Устройство среды Gymnasium: что должен реализовать класс gym.Env."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, boxstyle="round,pad=0.01,rounding_size=0.03",
                                linewidth=2, edgecolor=C_ENV, facecolor="white"))
    ax.text(0.5, 0.88, "class MyEnv(gym.Env)", ha="center", va="center", fontsize=15, weight="bold", color=C_ENV,
            family="monospace")
    rows = [
        ("observation_space", "что видит агент: Discrete(16), Box(low, high, shape), Dict, ...", C_AGENT),
        ("action_space", "что агент может делать: Discrete(4), Box(-1, 1, (2,)), ...", C_AGENT),
        ("reset(seed)", "начать эпизод, вернуть obs, info; seed задаёт self.np_random", C_ENV),
        ("step(action)", "один шаг: вернуть obs, reward, terminated, truncated, info", C_ENV),
        ("render() / close()", "картинка или текст для человека; освободить ресурсы", C_GREY),
    ]
    y = 0.72
    for name, desc, color in rows:
        _box(ax, (0.06, y - 0.05), 0.40, 0.10, name, color, fontsize=10.5, radius=0.02)
        ax.text(0.49, y, desc, ha="left", va="center", fontsize=11, color=C_TEXT)
        y -= 0.145
    fig.tight_layout()
    fig.savefig(OUT / "env_anatomy.png", dpi=DPI)
    plt.close(fig)


def wrapper_onion():
    """Обёртки: среда как луковица. gym.make сам добавляет три стандартных слоя."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    layers = [
        ("RecordEpisodeStatistics", C_PURPLE, "ваша обёртка: складывает return и длину эпизода в info"),
        ("TimeLimit(max_episode_steps=500)", C_REWARD, "обрывает эпизод: truncated=True"),
        ("OrderEnforcing", C_TEAL, "ругается, если step() вызван до reset()"),
        ("PassiveEnvChecker", C_TEAL, "проверяет типы наблюдений и наград"),
        ("CartPoleEnv", C_ENV, "сама среда: env.unwrapped"),
    ]
    n = len(layers)
    for i, (name, color, _) in enumerate(layers):
        pad = 0.035 * i
        ax.add_patch(FancyBboxPatch((0.04 + pad, 0.06 + pad * 1.6), 0.50 - 2 * pad, 0.86 - pad * 3.2,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    linewidth=2.5, edgecolor=color, facecolor="white"))
        top = 0.06 + pad * 1.6 + 0.86 - pad * 3.2
        ax.text(0.29, top - 0.035, name, ha="center", va="center", fontsize=9.5,
                color=color, weight="bold", family="monospace")
    ax.text(0.29, 0.42, "obs, reward,\nterminated, truncated", ha="center", va="center", fontsize=9, color=C_GREY)
    y = 0.84
    for name, color, desc in layers:
        ax.text(0.58, y, "■", color=color, fontsize=14, va="center")
        ax.text(0.61, y, desc, fontsize=11, va="center", color=C_TEXT)
        y -= 0.13
    ax.text(0.58, 0.14, "gym.make(...) добавляет три нижних слоя сам,\nостальные — вы, снаружи внутрь",
            fontsize=10.5, color=C_GREY, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "wrapper_onion.png", dpi=DPI)
    plt.close(fig)


def reward_types():
    """Разреженная и плотная награда на одной траектории."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    t = np.arange(0, 21)
    sparse = np.zeros_like(t, dtype=float); sparse[-1] = 1.0
    path = np.array([20, 19, 18, 17, 16, 15, 16, 17, 16, 15, 14, 13, 12, 11, 10, 8, 6, 4, 3, 1, 0])
    dist = path / 20                                # расстояние до цели (с небольшим крюком)
    dense = np.append(dist[:-1] - dist[1:], 0.0) - 0.02   # приближение к цели минус штраф за шаг
    dense[-1] += 1.0
    for ax, y, title, color in ((axes[0], sparse, "разреженная: 0, 0, 0, ..., 1", C_REWARD),
                                (axes[1], dense, "плотная: подсказка на каждом шаге", C_AGENT)):
        ax.bar(t, y, color=color, width=0.8)
        ax.set_title(title, fontsize=12); ax.set_xlabel("шаг эпизода"); ax.set_ylabel("награда")
        ax.axhline(0, color=C_GREY, lw=0.8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "reward_types.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (agent_env_loop, ml_paradigms, rl_timeline, rl_origins, rl_taxonomy, mdp_gridworld, course_map, markov_chain, cem_loop,
               env_anatomy, wrapper_onion, reward_types):
        fn()
        print("saved", fn.__name__)
