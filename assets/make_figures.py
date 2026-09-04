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
    _box(ax, (0.03, 0.62), 0.2, 0.1, "Табличные\n(недели 1-3)", C_GREY, fontsize=11)
    _box(ax, (0.27, 0.62), 0.28, 0.1, "Model-free deep RL", C_AGENT, fontsize=12)
    _box(ax, (0.59, 0.62), 0.18, 0.1, "Model-based\n(неделя 9)", C_ENV, fontsize=11)
    _box(ax, (0.8, 0.62), 0.18, 0.1, "Offline / imitation\n(неделя 11)", C_PURPLE, fontsize=11)
    for x in (0.13, 0.41, 0.68, 0.89):
        _arrow(ax, (0.5, 0.86), (x, 0.72), C_GREY, lw=1.5)

    # табличные
    ax.text(0.13, 0.55, "бандиты, DP,\nMonte Carlo, TD,\nSARSA, Q-learning", ha="center", va="top",
            fontsize=10, color=C_TEXT)
    # model-based
    ax.text(0.68, 0.55, "Dyna, MBPO,\nMuZero, Dreamer", ha="center", va="top", fontsize=10, color=C_TEXT)
    # offline
    ax.text(0.89, 0.55, "Behavior Cloning,\nDAgger, CQL", ha="center", va="top", fontsize=10, color=C_TEXT)

    # уровень 2 под model-free
    _box(ax, (0.2, 0.36), 0.14, 0.09, "Value-based\n(нед. 4-5)", C_TEAL, fontsize=10, text_color=C_TEXT)
    _box(ax, (0.34, 0.36), 0.14, 0.09, "Policy-based\n(нед. 6-7)", C_TEAL, fontsize=10, text_color=C_TEXT)
    _box(ax, (0.48, 0.36), 0.14, 0.09, "Actor-Critic\n(нед. 6-8)", C_TEAL, fontsize=10, text_color=C_TEXT)
    for x in (0.27, 0.41, 0.55):
        _arrow(ax, (0.41, 0.62), (x, 0.45), C_GREY, lw=1.5)
    ax.text(0.27, 0.32, "DQN, Double DQN,\nDueling, Rainbow", ha="center", va="top", fontsize=9.5, color=C_TEXT)
    ax.text(0.41, 0.32, "REINFORCE,\nTRPO, PPO", ha="center", va="top", fontsize=9.5, color=C_TEXT)
    ax.text(0.55, 0.32, "A2C, DDPG,\nTD3, SAC", ha="center", va="top", fontsize=9.5, color=C_TEXT)

    # нижняя полоса: расширения
    _box(ax, (0.03, 0.04), 0.95, 0.12,
         "Расширения: exploration (нед. 10)  ·  multi-agent (нед. 12)  ·  distributional / hierarchical / meta-RL, "
         "POMDP (нед. 13)  ·  RLHF, DPO для LLM (нед. 14)",
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
        ("Основы", C_GREY, ["1. Введение в RL", "2. Бандиты, MDP,\n    динамическое\n    программирование",
                            "3. Monte Carlo и TD", "4. Аппроксимация\n    функций, TD(λ)"]),
        ("Deep RL", C_AGENT, ["5. Deep Q-Learning", "6. Policy Gradient", "7. TRPO и PPO",
                              "8. DDPG, TD3, SAC"]),
        ("Расширения", C_ENV, ["9. Model-based RL", "10. Exploration", "11. Imitation и Offline RL",
                               "12. Multi-agent RL"]),
        ("Фронтир", C_REWARD, ["13. Distributional,\n      hierarchical, meta-RL", "14. RLHF и DPO для LLM",
                               "15. Фронтир RL,\n      защита проектов"]),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w = 0.225
    gap = 0.03
    for k, (title, color, items) in enumerate(blocks):
        x = 0.01 + k * (w + gap)
        _box(ax, (x, 0.84), w, 0.13, title, color, fontsize=14)
        ax.add_patch(FancyBboxPatch((x, 0.04), w, 0.74, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        y = 0.68
        for item in items:
            ax.text(x + 0.015, y, item, ha="left", va="center", fontsize=11.5, color=C_TEXT, linespacing=1.4)
            y -= 0.17
        if k < len(blocks) - 1:
            ax.text(x + w + gap / 2, 0.905, "→", ha="center", va="center", fontsize=18, color=C_GREY)
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


if __name__ == "__main__":
    for fn in (agent_env_loop, ml_paradigms, rl_timeline, rl_origins, rl_taxonomy, mdp_gridworld, course_map):
        fn()
        print("saved", fn.__name__)
