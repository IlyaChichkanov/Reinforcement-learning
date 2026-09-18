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


def _grid_world(ax, cell=1.0, x0=0.0, y0=0.0, numbers=False):
    """Карта клетчатого мира 3 x 4: стена, выход +1, яма -1, старт. Возвращает функцию «центр клетки»."""
    layout = ["...G", ".#.X", "S..."]
    colors = {".": "#ffffff", "S": "#f2f2f2", "#": "#555555", "G": "#a9dfa9", "X": "#f3a9a9"}
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            x, y = x0 + c * cell, y0 + (len(layout) - 1 - r) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=colors[ch], edgecolor="black", lw=1.2))
            if ch == "G":
                ax.text(x + cell / 2, y + cell / 2, "+1", ha="center", va="center", fontsize=17, weight="bold", color="#2f6b2f")
            if ch == "X":
                ax.text(x + cell / 2, y + cell / 2, "−1", ha="center", va="center", fontsize=17, weight="bold", color="#9b2f2f")
            if ch == "S":
                ax.text(x + cell / 2, y + cell / 2, "старт", ha="center", va="center", fontsize=10, color=C_GREY)
            if numbers and ch != "#":
                ax.text(x + 0.07 * cell, y + 0.88 * cell, str(r * 4 + c), ha="left", va="center", fontsize=9, color=C_GREY)
    return lambda r, c: (x0 + (c + 0.5) * cell, y0 + (len(layout) - 0.5 - r) * cell)


def mdp_gridworld():
    """Клетчатый мир: состояние — клетка, действие — направление, ветер сносит вбок."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-1.9, 3.3); ax.axis("off"); ax.set_aspect("equal")
    center = _grid_world(ax)
    x, y = center(2, 1)                                       # агент в нижнем ряду
    ax.scatter([x], [y - 0.16], s=340, color=C_AGENT, zorder=3)
    for dx, dy in [(0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)]:
        _arrow(ax, (x, y - 0.16), (x + dx, y - 0.16 + dy), C_AGENT, lw=1.8)
    ax.text(x + 0.40, y + 0.12, "$a$", fontsize=14, color=C_AGENT, weight="bold")
    ax.text(2.0, -0.55, "Состояние — клетка, действие — одно из четырёх направлений.", ha="center", fontsize=11.5, color=C_TEXT)
    ax.text(2.0, -0.95, "Награда: −0.04 за каждый шаг (чтобы не гулять вечно), +1 за выход, −1 за яму.", ha="center", fontsize=11.5, color=C_TEXT)
    ax.text(2.0, -1.35, "Ветер: с вероятностью 0.8 идём куда хотели, по 0.1 — вбок;\nв стену или за край — остаёмся на месте.", ha="center", fontsize=11.5, color=C_TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "mdp_gridworld.png", dpi=DPI)
    plt.close(fig)


def gridworld_rules():
    """Правила мира лекции 2 крупно: карта с номерами клеток, легенда и веер ветра."""
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(-0.5, 5.0); ax.axis("off"); ax.set_aspect("equal")
    _grid_world(ax, cell=1.15, x0=0.2, y0=0.9, numbers=True)
    ax.text(2.5, 4.65, "Мир: 3 × 4 клетки", ha="center", fontsize=13.5, weight="bold", color=C_TEXT)
    ax.text(2.5, 0.45, "эпизод кончается в зелёной или красной клетке", ha="center", fontsize=11, color=C_GREY)
    lines = [
        ("Состояние", "номер клетки, 0…11 (клетка 5 — стена)"),
        ("Действие", "← ↓ → ↑ — попытка шагнуть в эту сторону"),
        ("Награда", "−0.04 за шаг, +1 за выход, −1 за яму"),
        ("Ветер", "0.8 — куда просили, по 0.1 — вбок"),
        ("Стена и край", "шаг в них оставляет агента на месте"),
    ]
    for i, (name, text) in enumerate(lines):
        y = 4.6 - 0.6 * i
        ax.text(5.0, y, name, ha="left", va="center", fontsize=12.5, weight="bold", color=C_AGENT)
        ax.text(7.05, y, text, ha="left", va="center", fontsize=12.5, color=C_TEXT)
    bx, by = 8.6, 0.2
    ax.text(bx, by + 1.38, "просим «вверх» — а получается так:", ha="center", fontsize=11.5, color=C_TEXT, weight="bold")
    ax.scatter([bx], [by], s=300, color=C_AGENT, zorder=3)
    for dx, dy, p, ha, off in [(0.0, 1.05, "0.8", "left", (0.22, -0.2)), (-1.25, 0.3, "0.1", "right", (-0.18, 0.06)),
                               (1.25, 0.3, "0.1", "left", (0.18, 0.06))]:
        _arrow(ax, (bx, by + 0.12), (bx + dx, by + 0.12 + dy), C_ENV, lw=2.4)
        ax.text(bx + dx + off[0], by + 0.12 + dy + off[1], p, ha=ha, va="center", fontsize=12.5, color=C_ENV, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "gridworld_rules.png", dpi=DPI)
    plt.close(fig)


def v_vs_q():
    """Разница между V и Q на том же лабиринте: одно число в клетке против четырёх."""
    import sys
    sys.path.insert(0, str(OUT.parent / "02-environments" / "lecture"))
    from gridworld import GridWorld, mdp_matrices, solve_q_star, ARROWS          # noqa: E402
    from gridworld_plots import plot_values, plot_q                              # noqa: E402

    env = GridWorld()
    P, R = mdp_matrices(env)
    Q = solve_q_star(P, R)
    V = Q.max(axis=1)
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.22, wspace=0.12)
    ax_v, ax_q = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    plot_values(V, env, ax=ax_v, title="$V(s)$ — насколько хороша клетка")
    plot_q(Q, env, ax=ax_q, title="$Q(s, a)$ — насколько хорош каждый шаг из клетки")
    cell = 9
    best = int(Q[cell].argmax())
    ax_text = fig.add_subplot(gs[1, :]); ax_text.axis("off")
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.text(0.25, 0.92, "одно число на клетку", ha="center", fontsize=12, color=C_AGENT, weight="bold")
    ax_text.text(0.25, 0.55, "«сколько соберу, если я здесь\nи дальше играю хорошо»", ha="center", fontsize=11.5, color=C_TEXT)
    ax_text.text(0.75, 0.92, "четыре числа на клетку", ha="center", fontsize=12, color=C_AGENT, weight="bold")
    ax_text.text(0.75, 0.55, "«сколько соберу, если сделаю именно это действие,\nа дальше играю хорошо»", ha="center",
                 fontsize=11.5, color=C_TEXT)
    link = (f"Связь: $V(s) = \\max_a Q(s,a)$. В клетке {cell}: "
            + ", ".join(f"{ARROWS[a]} {Q[cell, a]:+.2f}" for a in range(4))
            + f"   →   лучшее {ARROWS[best]} {Q[cell, best]:+.2f} = $V$({cell})")
    ax_text.text(0.5, 0.22, link, ha="center", fontsize=12, color=C_TEXT)
    ax_text.text(0.5, -0.02, "Из $Q$ стратегия получается сразу: выбирай наибольшее число. "
                            "Из $V$ — только если знаешь, куда ведут действия.",
                 ha="center", fontsize=12.5, color=C_ACCENT, weight="bold")
    fig.savefig(OUT / "v_vs_q.png", dpi=DPI, bbox_inches="tight", pad_inches=0.2)
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



def observation_vs_state():
    """Состояние и наблюдение: два разных состояния могут выглядеть одинаково."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # два состояния слева
    _box(ax, (0.04, 0.60), 0.24, 0.26, "Состояние A\nшест падает вправо,\nугол +3°", C_ENV, fontsize=10.5, radius=0.02)
    _box(ax, (0.04, 0.14), 0.24, 0.26, "Состояние B\nшест возвращается,\nугол +3°", C_ENV, fontsize=10.5, radius=0.02)
    # датчик
    _box(ax, (0.40, 0.37), 0.20, 0.26, "Датчик\n(только угол)", C_GREY, fontsize=11, radius=0.02)
    _arrow(ax, (0.28, 0.73), (0.40, 0.55), C_GREY, lw=2)
    _arrow(ax, (0.28, 0.27), (0.40, 0.45), C_GREY, lw=2)
    # наблюдение справа
    _box(ax, (0.72, 0.37), 0.24, 0.26, "Наблюдение\n«угол = +3°»", C_AGENT, fontsize=11, radius=0.02)
    _arrow(ax, (0.60, 0.50), (0.72, 0.50), C_AGENT, lw=2.5)
    ax.text(0.84, 0.28, "одно и то же —\nа действия нужны разные", ha="center", va="top", fontsize=10.5,
            color=C_ACCENT, weight="bold")
    ax.text(0.5, 0.06, "Лекарства: добавить в наблюдение историю (стек кадров, разность) или дать агенту память",
            ha="center", va="center", fontsize=10.5, color=C_TEXT)
    ax.text(0.5, 0.95, "Наблюдение ≠ состояние: частичная наблюдаемость (POMDP)", ha="center", va="center",
            fontsize=13, weight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "observation_vs_state.png", dpi=DPI)
    plt.close(fig)


def action_spaces():
    """Четыре типа пространств действий с примерами."""
    cards = [
        ("Discrete(n)", C_AGENT, "одно из n действий",
         "4 направления в GridWorld\n18 кнопок Atari\nход в шахматах (с маской)"),
        ("Box(low, high, shape)", C_ENV, "вектор вещественных чисел",
         "момент на суставе робота\nугол руля и газ\nобъём заявки на бирже"),
        ("MultiDiscrete([n1, n2])", C_PURPLE, "несколько дискретных сразу",
         "кнопка + направление\nв Dota: «что» + «куда»"),
        ("Dict / Tuple", C_REWARD, "составное действие",
         "{move: Discrete(4),\n fire: Discrete(2)}"),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, gap = 0.225, 0.028
    for k, (name, color, sub, examples) in enumerate(cards):
        x = 0.01 + k * (w + gap)
        _box(ax, (x, 0.74), w, 0.2, name, color, fontsize=12, radius=0.02)
        ax.add_patch(FancyBboxPatch((x, 0.05), w, 0.64, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, 0.60, sub, ha="center", va="center", fontsize=11, color=color, weight="bold")
        ax.text(x + w / 2, 0.42, examples, ha="center", va="center", fontsize=10.5, color=C_TEXT, linespacing=1.5)
        ax.text(x + w / 2, 0.12, ["табличные, DQN, PPO", "policy gradient, DDPG/TD3/SAC, PPO",
                                  "чаще всего сводят к Discrete", "сводят к одному из двух"][k],
                ha="center", va="center", fontsize=9.5, color=C_GREY, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "action_spaces.png", dpi=DPI)
    plt.close(fig)


def policy_types():
    """Политика: таблица или функция, детерминированная или стохастическая."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # слева: табличная политика
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "Табличная: π[s, a]", ha="center", va="center", fontsize=13, weight="bold", color=C_AGENT)
    rows = [("s = 0", [0.7, 0.1, 0.1, 0.1]), ("s = 1", [0.0, 0.0, 1.0, 0.0]), ("s = 2", [0.25, 0.25, 0.25, 0.25]),
            ("...", None)]
    ax.text(0.30, 0.82, "←", ha="center", fontsize=12); ax.text(0.45, 0.82, "↓", ha="center", fontsize=12)
    ax.text(0.60, 0.82, "→", ha="center", fontsize=12); ax.text(0.75, 0.82, "↑", ha="center", fontsize=12)
    y = 0.70
    for name, probs in rows:
        ax.text(0.12, y, name, ha="center", va="center", fontsize=11, family="monospace", color=C_TEXT)
        if probs is not None:
            for j, p in enumerate(probs):
                ax.add_patch(Rectangle((0.24 + 0.15 * j, y - 0.05), 0.12, 0.10, facecolor=C_AGENT, alpha=0.15 + 0.85 * p,
                                       edgecolor="white"))
                ax.text(0.30 + 0.15 * j, y, f"{p:.2f}", ha="center", va="center", fontsize=9.5,
                        color="white" if p > 0.5 else C_TEXT)
        y -= 0.15
    ax.text(0.5, 0.12, "s = 1: детерминированная строка\ns = 2: ещё ничего не выучено", ha="center", va="center",
            fontsize=10, color=C_GREY)
    # справа: параметрическая
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "Параметрическая: π_θ(a | s) = softmax(f_θ(s))", ha="center", va="center", fontsize=13,
            weight="bold", color=C_ENV)
    _box(ax, (0.02, 0.42), 0.20, 0.16, "s\n(вектор,\nкартинка)", C_GREY, fontsize=9.5, radius=0.02)
    _box(ax, (0.32, 0.36), 0.30, 0.28, "f_θ\nлинейная модель\nили нейросеть", C_ENV, fontsize=10.5, radius=0.02)
    _box(ax, (0.72, 0.42), 0.26, 0.16, "softmax →\nπ(a | s)", C_AGENT, fontsize=10.5, radius=0.02)
    _arrow(ax, (0.22, 0.50), (0.32, 0.50), C_GREY, lw=2)
    _arrow(ax, (0.62, 0.50), (0.72, 0.50), C_GREY, lw=2)
    ax.text(0.5, 0.20, "обучаем веса θ градиентом (неделя 5);\nобобщает на невиденные состояния",
            ha="center", va="center", fontsize=10, color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "policy_types.png", dpi=DPI)
    plt.close(fig)



def return_recursion():
    """Return одного эпизода клетчатого мира: награды, веса γ^t и подсчёт G_t с конца."""
    gamma = 0.95
    cells = [8, 4, 0, 1, 2, 3]
    rewards = [-0.04, -0.04, -0.04, -0.04, 0.96]          # последний шаг: −0.04 за шаг и +1 за выход
    G = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        G[t] = rewards[t] + (gamma * G[t + 1] if t + 1 < len(rewards) else 0.0)
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(cells))
    for i, (x, c) in enumerate(zip(xs, cells)):
        color = C_ENV if c == 3 else C_AGENT
        ax.add_patch(FancyBboxPatch((x - 0.045, 0.74), 0.09, 0.16, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=0, facecolor=color))
        ax.text(x, 0.82, f"клетка\n{c}" + (" (+1)" if c == 3 else ""), ha="center", va="center",
                fontsize=9.5, color="white", weight="bold")
        if i < len(rewards):
            _arrow(ax, (x + 0.05, 0.82), (xs[i + 1] - 0.05, 0.82), C_GREY, lw=1.5)
            ax.text((x + xs[i + 1]) / 2, 0.67, f"$r_{i}$ = {rewards[i]:+.2f}", ha="center", va="center",
                    fontsize=10.5, color=C_REWARD if rewards[i] > 0 else C_GREY,
                    weight="bold" if rewards[i] > 0 else "normal")
            ax.text((x + xs[i + 1]) / 2, 0.58, f"вес $\\gamma^{i}$ = {gamma ** i:.2f}", ha="center", va="center",
                    fontsize=8.5, color=C_GREY)
    ax.text(0.5, 0.96, "один эпизод: из старта вверх и направо к выходу", ha="center", va="center",
            fontsize=12, color=C_TEXT)
    ax.plot([0.02, 0.98], [0.49, 0.49], color=C_GREY, lw=0.8, ls=":")
    for i in range(len(rewards)):
        xm = (xs[i] + xs[i + 1]) / 2
        ax.add_patch(FancyBboxPatch((xm - 0.055, 0.29), 0.11, 0.12, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=0, facecolor=C_ACCENT if i == 0 else C_LIGHT))
        ax.text(xm, 0.35, f"$G_{i}$ = {G[i]:+.3f}", ha="center", va="center", fontsize=10,
                color="white" if i == 0 else C_TEXT, weight="bold")
        if i + 1 < len(rewards):
            xn = (xs[i + 1] + xs[i + 2]) / 2
            _arrow(ax, (xn - 0.06, 0.35), (xm + 0.06, 0.35), C_ACCENT, lw=1.3)
            ax.text((xm + xn) / 2, 0.23, f"× $\\gamma$ + $r_{i}$", ha="center", va="center", fontsize=8.5, color=C_ACCENT)
    ax.text(0.5, 0.11, "return $G_t$ — сумма наград с шага $t$ с весами $\\gamma^k$. Считаем справа налево:  "
            "$G_4 = r_4 = 0.96$,  $G_3 = r_3 + \\gamma G_4 = 0.87$, …", ha="center", va="center", fontsize=11, color=C_TEXT)
    ax.text(0.5, 0.02, "в общем виде   $G_t = r_t + \\gamma\\, G_{t+1}$", ha="center", va="center",
            fontsize=13, color=C_ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "return_recursion.png", dpi=DPI)
    plt.close(fig)


def from_episodes_to_steps():
    """Мост от лекции 1: Cross-Entropy оценивает эпизод целиком, а хочется оценивать каждый шаг."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.text(3.0, 4.55, "Лекция 1: Cross-Entropy", ha="center", fontsize=13.5, weight="bold", color=C_TEXT)
    steps = [("сыграть 200\nэпизодов", C_AGENT), ("отобрать 30%\nлучших", C_ENV), ("повторять их\nдействия", C_PURPLE)]
    for i, (text, color) in enumerate(steps):
        _box(ax, (0.3 + i * 1.95, 3.2), 1.6, 0.95, text, color, fontsize=10.5, radius=0.02)
        if i < 2:
            _arrow(ax, (1.95 + i * 1.95, 3.67), (2.2 + i * 1.95, 3.67), C_GREY, lw=1.8)
    ax.plot([5.4, 5.7, 5.7, 1.1], [3.67, 3.67, 2.82, 2.82], color=C_GREY, lw=1.5)
    _arrow(ax, (1.1, 2.82), (1.1, 3.15), C_GREY, lw=1.5)
    ax.text(3.4, 2.62, "и так 25 раз", ha="center", fontsize=10.5, color=C_GREY)
    ax.text(3.0, 2.3, "оценка эпизода — одно число: дошёл или не дошёл", ha="center", fontsize=11, color=C_GREY)
    # правая часть: проблема
    ax.text(9.6, 4.55, "Что с этим не так", ha="center", fontsize=13.5, weight="bold", color=C_TEXT)
    n = 24
    for i in range(n):
        bad = i == 17
        ax.add_patch(Rectangle((6.5 + i * 0.26, 3.35), 0.22, 0.6, facecolor=C_ACCENT if bad else C_ENV,
                               edgecolor="none"))
    ax.text(6.5 + 17 * 0.26 + 0.11, 4.12, "одна ошибка", ha="center", fontsize=10, color=C_ACCENT, weight="bold")
    ax.text(9.6, 3.0, "эпизод = 100 шагов. Он «плохой» — выбрасываем целиком,\nвместе с 99 верными шагами",
            ha="center", fontsize=11, color=C_TEXT)
    _arrow(ax, (9.6, 2.5), (9.6, 1.85), C_ACCENT, lw=2.2)
    _box(ax, (6.3, 0.75), 6.6, 1.0, "Хотим оценивать не эпизод, а шаг:\nнасколько хорошо сделать это действие в этой клетке?",
         C_ACCENT, fontsize=12, radius=0.02)
    ax.text(3.0, 1.25, "сегодня: как такую оценку\nпосчитать и как из неё\nсразу получить стратегию",
            ha="center", fontsize=11.5, color=C_TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "from_episodes_to_steps.png", dpi=DPI)
    plt.close(fig)


def two_equations():
    """Два вопроса — два уравнения Беллмана: max по действиям против усреднения по стратегии."""
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis("off")
    blocks = [
        (0.3, C_ACCENT, "Вопрос 1: как ходить лучше всего?",
         "$Q^*(s,a) = \\mathbb{E}\\,[\\,r + \\gamma\\, \\max_{a'} Q^*(s',a')\\,]$",
         "$\\max$: дальше агент выберет\nсамое выгодное действие",
         "ответ: оптимальная стратегия\n$\\pi^*(s) = \\arg\\max_a Q^*(s,a)$"),
        (6.8, C_AGENT, "Вопрос 2: насколько хороша стратегия $\\pi$?",
         "$V^\\pi(s) = \\sum_a \\pi(a|s)\\, \\mathbb{E}\\,[\\,r + \\gamma\\, V^\\pi(s')\\,]$",
         "$\\sum_a \\pi(a|s)$: дальше агент\nсделает то, что велит $\\pi$",
         "ответ: оценка стратегии —\nсравнить две, улучшить одну"),
    ]
    for x, color, head, formula, note, answer in blocks:
        _box(ax, (x, 4.05), 5.9, 0.75, head, color, fontsize=12.5, radius=0.02)
        ax.add_patch(FancyBboxPatch((x, 2.85), 5.9, 1.0, boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor=C_LIGHT, linewidth=0))
        ax.text(x + 2.95, 3.35, formula, ha="center", va="center", fontsize=14, color=C_TEXT)
        ax.text(x + 2.95, 2.25, note, ha="center", va="center", fontsize=11, color=color)
        _arrow(ax, (x + 2.95, 1.85), (x + 2.95, 1.35), C_GREY, lw=1.8)
        ax.text(x + 2.95, 0.95, answer, ha="center", va="center", fontsize=11.5, color=C_TEXT)
    ax.text(6.5, 0.15, "Отличие ровно одно:   $\\max_a$   против   $\\sum_a \\pi(a|s)$", ha="center", va="center",
            fontsize=13, color=C_ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "two_equations.png", dpi=DPI)
    plt.close(fig)


def q_choice():
    """Принцип оптимальности Беллмана на пальцах: известны Q*(s, a) для трёх действий — выбираем наибольшее."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.scatter([0.16], [0.56], s=2600, color=C_AGENT, zorder=3)
    ax.text(0.16, 0.56, "s", ha="center", va="center", fontsize=18, color="white", weight="bold", zorder=4)
    items = [(0.86, "a = 0", "$Q^*(s, 0) = +3$", "больше +3 отсюда не набрать", C_GREY),
             (0.56, "a = 1", "$Q^*(s, 1) = +10$", "есть стратегия, которая набирает +10", C_ACCENT),
             (0.26, "a = 2", "$Q^*(s, 2) = -1$", "в среднем потеряем 1", C_GREY)]
    for y, name, q, note, color in items:
        _arrow(ax, (0.21, 0.56 + (y - 0.56) * 0.25), (0.36, y), color, lw=3.5 if color == C_ACCENT else 1.8)
        ax.text(0.28, 0.56 + (y - 0.56) * 0.62 + 0.04, name, ha="center", va="center", fontsize=10, color=color)
        _box(ax, (0.37, y - 0.08), 0.24, 0.16, q, color, fontsize=13, radius=0.02)
        ax.text(0.64, y, note, ha="left", va="center", fontsize=10.5, color=C_TEXT if color == C_ACCENT else C_GREY)
    ax.text(0.5, 0.0, "жадный выбор $\\pi^*(s) = \\arg\\max_a Q^*(s, a)$: здесь $a = 1$.\nЛучше по определению $Q^*$ не бывает.",
            ha="center", va="bottom", fontsize=11, color=C_ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "q_choice.png", dpi=DPI)
    plt.close(fig)


def lottery_mdp():
    """MDP «Колобок и лотерея» из учебника ШАД: одно состояние, два действия, редкий большой выигрыш."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.scatter([0.16], [0.60], s=5200, color=C_AGENT, zorder=3)
    ax.text(0.16, 0.60, "s:\nу киоска", ha="center", va="center", fontsize=12, color="white", weight="bold", zorder=4)
    # действие «купить билет» — точка, из которой среда выбирает исход
    ax.scatter([0.46], [0.60], s=380, color=C_TEXT, zorder=3)
    _arrow(ax, (0.215, 0.60), (0.435, 0.60), C_REWARD, lw=3)
    ax.text(0.33, 0.68, "«купить билет»\n$r = -10$", ha="center", va="center", fontsize=11, color=C_REWARD, weight="bold")
    ax.add_patch(FancyArrowPatch((0.46, 0.635), (0.185, 0.675), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=C_ENV,
                                 connectionstyle="arc3,rad=0.6"))
    ax.text(0.34, 0.92, "с вероятностью $p$ — проиграл,\nснова у киоска ($s' = s$)", ha="center", va="center", fontsize=10, color=C_ENV)
    _arrow(ax, (0.475, 0.575), (0.66, 0.40), C_ENV, lw=2.2)
    ax.text(0.62, 0.55, "с вероятностью $1-p$ —\nвыигрыш", ha="center", va="center", fontsize=10, color=C_ENV)
    _box(ax, (0.67, 0.30), 0.28, 0.16, "выигрыш: +1000\n(эпизод окончен)", C_ENV, fontsize=11.5, radius=0.02)
    # действие «не покупать»
    _arrow(ax, (0.16, 0.48), (0.16, 0.24), C_GREY, lw=2.0)
    ax.text(0.175, 0.36, "«не покупать»: $r = 0$", ha="left", va="center", fontsize=10.5, color=C_GREY, weight="bold")
    _box(ax, (0.02, 0.06), 0.28, 0.14, "ушёл домой\n(эпизод окончен)", C_GREY, fontsize=11, radius=0.02)
    ax.text(0.66, 0.16, "$Q^*(s, \\text{купить}) = -10 + \\gamma\\,[\\,p \\cdot \\max(Q^*(s, \\text{купить}),\\, 0) + (1-p)\\cdot 1000\\,]$",
            ha="center", va="center", fontsize=10.5, color=C_ACCENT, weight="bold")
    ax.text(0.66, 0.06, "($Q^*(s, \\text{не покупать}) = 0$; при $\\gamma = 1$ и $p = 0.99$ решение $Q^*(s, \\text{купить}) = 0$)",
            ha="center", va="center", fontsize=9.5, color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "lottery_mdp.png", dpi=DPI)
    plt.close(fig)


def vi_update():
    """Одна строка кода value iteration и формула, которую она считает, часть в часть."""
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")
    C_R, C_G, C_P = C_REWARD, C_PURPLE, C_ENV
    ax.text(6.5, 5.05, "Одна итерация: формула и код — это одно и то же", ha="center", fontsize=14,
            weight="bold", color=C_TEXT)

    # формула по частям, чтобы куски стояли ровно над кусками кода
    parts_formula = [(1.05, "$Q_{k+1}(s,a)\;=$", C_TEXT), (3.45, "$r(s,a)$", C_R),
                     (5.0, "$+\;\\gamma$", C_G), (6.55, "$\\sum_{s'} p(s'|s,a)\\, V_k(s')$", C_P)]
    parts_code = [(1.05, "Q_new", C_TEXT), (2.55, "=", C_TEXT), (3.45, "R", C_R),
                  (4.55, "+", C_TEXT), (5.0, "GAMMA", C_G), (6.0, "*", C_TEXT), (6.75, "(P @ V)", C_P)]
    for x, text, color in parts_formula:
        ax.text(x, 4.15, text, ha="left", va="center", fontsize=16, color=color)
    ax.add_patch(FancyBboxPatch((0.7, 2.85), 11.6, 0.85, boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=C_LIGHT, linewidth=0))
    for x, text, color in parts_code:
        ax.text(x, 3.28, text, ha="left", va="center", fontsize=15, color=color,
                family="monospace", weight="bold" if color != C_TEXT else "normal")
    for x0, x1 in [(3.45, 3.45), (5.0, 5.0), (6.55, 6.75)]:
        ax.plot([x0 + 0.15, x1 + 0.15], [3.95, 3.75], color=C_GREY, lw=1, ls=":")

    ax.text(1.0, 2.35, "$V_k$ — лучшее, что обещает текущее приближение:", ha="left", fontsize=12, color=C_TEXT)
    ax.text(8.0, 2.35, "V = Q.max(axis=1)", ha="left", fontsize=13, color=C_TEXT, family="monospace")
    ax.text(8.0, 1.95, "(максимум по действиям — по второй оси)", ha="left", fontsize=10, color=C_GREY)

    rows = [("R", "(12, 4)", "награда за шаг для каждой пары «клетка, действие»", C_R),
            ("V", "(12,)", "по одному числу на клетку", C_P),
            ("P", "(12, 4, 12)", "вероятности: куда попадём из каждой пары", C_P),
            ("P @ V", "(12, 4)", "средняя ценность следующей клетки", C_P),
            ("Q_new", "(12, 4)", "48 чисел — всё приближение целиком", C_TEXT)]
    ax.text(1.0, 1.45, "формы массивов", ha="left", fontsize=12, weight="bold", color=C_AGENT)
    for i, (name, shape, text, color) in enumerate(rows):
        y = 1.05 - 0.26 * i
        ax.text(1.0, y, name, ha="left", va="center", fontsize=11.5, color=color, family="monospace", weight="bold")
        ax.text(2.6, y, shape, ha="left", va="center", fontsize=11.5, color=C_TEXT, family="monospace")
        ax.text(4.6, y, text, ha="left", va="center", fontsize=11.5, color=C_GREY)
    ax.text(12.3, 0.55, "одна строка пересчитывает\nвсе 48 чисел сразу", ha="right", va="center",
            fontsize=12.5, color=C_ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "vi_update.png", dpi=DPI)
    plt.close(fig)


def bellman_family():
    """Одно уравнение — весь курс: какие методы что подставляют в уравнение Беллмана."""
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.30, 0.42), 0.40, 0.18, "Уравнение Беллмана\n$Q(s,a) = \\mathbb{E}\\,[\\,r + \\gamma\\, Q(s', a')\\,]$", C_TEXT, fontsize=13)
    items = [
        ((0.02, 0.76), "Динамическое\nпрограммирование (нед. 3)", C_ENV, "знаем $p(s'|s,a)$ и $r$: решаем\nуравнение простой итерацией"),
        ((0.36, 0.76), "Monte-Carlo (нед. 3)", C_AGENT, "вместо $\\mathbb{E}$ — среднее\nпо целым эпизодам"),
        ((0.70, 0.76), "Q-learning, SARSA, TD\n(сегодня и нед. 3)", C_AGENT, "вместо $\\mathbb{E}$ — один переход $(s, a, r, s')$:\nсдвигаемся к таргету $r + \\gamma Q(s', a')$"),
        ((0.02, 0.06), "DQN (нед. 6)", C_PURPLE, "вместо таблицы $Q$ — нейросеть,\nневязка уравнения — функция потерь"),
        ((0.36, 0.06), "Actor-Critic, PPO\n(нед. 8–9)", C_PURPLE, "критик учит $Q$ или $V$ по Беллману,\nактор улучшает стратегию"),
        ((0.70, 0.06), "Model-based (нед. 11–12)", C_REWARD, "учим $p(s'|s,a)$ и $r$ по данным,\nдальше как в DP"),
    ]
    for (x, y), name, color, desc in items:
        _box(ax, (x, y), 0.28, 0.12, name, color, fontsize=11, radius=0.02)
        ax.text(x + 0.14, y - 0.035 if y > 0.5 else y + 0.155, desc, ha="center", va="top" if y > 0.5 else "bottom",
                fontsize=9.5, color=C_TEXT)
        cx, cy = x + 0.14, (y if y > 0.5 else y + 0.12)
        _arrow(ax, (0.5, 0.60 if y > 0.5 else 0.42), (cx, cy + (0 if y > 0.5 else 0.0)), C_GREY, lw=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "bellman_family.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (agent_env_loop, ml_paradigms, rl_timeline, rl_origins, rl_taxonomy, mdp_gridworld, course_map, markov_chain, cem_loop,
               env_anatomy, wrapper_onion, reward_types, observation_vs_state, action_spaces, policy_types,
               return_recursion, gridworld_rules, from_episodes_to_steps, q_choice, v_vs_q, two_equations, vi_update, lottery_mdp, bellman_family):
        fn()
        print("saved", fn.__name__)
