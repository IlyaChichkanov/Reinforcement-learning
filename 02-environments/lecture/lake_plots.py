"""Служебный код лекции 2: картинки на карте озера (Frozen Lake 4×4).

Функции draw_lake / draw_policy / plot_values повторяют лекцию 1; новые — draw_transitions, plot_q,
draw_backup, plot_q_updates. Содержательного кода здесь нет, читать не обязательно.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ARROWS = "←↓→↑"                      # так пронумерованы действия в Frozen Lake: 0=←, 1=↓, 2=→, 3=↑
ACTION_NAMES = ["влево", "вниз", "вправо", "вверх"]
CELL_COLORS = {"S": "#f2f2f2", "F": "#ffffff", "H": "#a8c8ec", "G": "#a9dfa9"}
C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"


def lake_desc(env):
    """Карта озера как список строк: ['SFFF', 'FHFH', 'FFFH', 'HFFG']."""
    return ["".join(c.decode() for c in row) for row in env.unwrapped.desc]


def draw_lake(desc, ax=None, size=3.0, numbers=False, letters="SHG"):
    """Пустая карта озера. numbers=True — подписать номера клеток; letters — какие буквы карты подписывать."""
    nrow, ncol = len(desc), len(desc[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(size, size * nrow / ncol))
    ax.set_xlim(0, ncol); ax.set_ylim(nrow, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for r in range(nrow):
        for c in range(ncol):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=CELL_COLORS[desc[r][c]], edgecolor="k", lw=0.6))
            if desc[r][c] in letters:
                ax.text(c + 0.5, r + 0.82, desc[r][c], ha="center", va="center", fontsize=8, color="#666")
            if numbers:
                ax.text(c + 0.08, r + 0.2, str(r * ncol + c), ha="left", va="center", fontsize=7, color="#999")
    return ax


def draw_policy(policy, desc, ax=None, title="", size=3.0):
    """Стрелка — самое вероятное действие, насыщенность — его вероятность; «·» — строка почти равномерная."""
    ax = draw_lake(desc, ax, size)
    ncol = len(desc[0])
    for s, p in enumerate(policy):
        r, c = divmod(s, ncol)
        if desc[r][c] in "HG":
            continue
        a, top2 = int(np.argmax(p)), np.sort(p)[-2:]
        if top2[1] - top2[0] < 0.05:
            ax.text(c + 0.5, r + 0.5, "·", ha="center", va="center", fontsize=18, color="grey")
        else:
            ax.text(c + 0.5, r + 0.5, ARROWS[a], ha="center", va="center", fontsize=20, alpha=0.2 + 0.8 * float(p[a]))
    ax.set_title(title, fontsize=10)
    return ax


def plot_values(values, desc, ax=None, title="", vmax=None, size=3.0):
    """Тепловая карта ценности клеток (проруби и цель — терминальные, у них ценность 0 по определению)."""
    nrow, ncol = len(desc), len(desc[0])
    ax = draw_lake(desc, ax, size)
    grid = np.asarray(values, float).reshape(nrow, ncol)
    vmax = vmax or max(float(np.nanmax(grid)), 1e-9)
    for r in range(nrow):
        for c in range(ncol):
            if desc[r][c] in "HG":
                continue
            if np.isnan(grid[r, c]):                     # клетку ни разу не посещали — оценки нет
                ax.text(c + 0.5, r + 0.5, "—", ha="center", va="center", fontsize=9, color="grey")
                continue
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=plt.cm.YlOrRd(0.85 * grid[r, c] / vmax), edgecolor="k", lw=0.6))
            ax.text(c + 0.5, r + 0.5, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    return ax


def draw_transitions(env, s, a, ax=None, title=None, size=3.4):
    """Куда среда может перевести агента из клетки s после действия a: стрелки с вероятностями (и наградой, если есть)."""
    desc = lake_desc(env)
    ncol = len(desc[0])
    ax = draw_lake(desc, ax, size, numbers=True)
    r0, c0 = divmod(s, ncol)
    ax.add_patch(Rectangle((c0, r0), 1, 1, facecolor="none", edgecolor=C_AGENT, lw=3))
    ax.text(c0 + 0.5, r0 + 0.45, ARROWS[a], ha="center", va="center", fontsize=22, color=C_AGENT, weight="bold")
    for prob, s_next, reward, _ in env.unwrapped.P[s][a]:
        r1, c1 = divmod(s_next, ncol)
        label = "1/3" if abs(prob - 1 / 3) < 1e-9 else f"{prob:g}"
        if reward:
            label += f", +{reward:g}"
        if s_next == s:                                            # шаг в стену: остаёмся на месте
            ax.text(c0 + 0.5, r0 + 0.85, f"остаёмся: {label}", ha="center", va="center", fontsize=7.5, color=C_ENV, weight="bold")
            continue
        p0, p1 = np.array([c0 + 0.5, r0 + 0.5]), np.array([c1 + 0.5, r1 + 0.5])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        ax.add_patch(FancyArrowPatch(p0 + 0.22 * d, p1 - 0.32 * d, arrowstyle="-|>", mutation_scale=18, lw=2.4, color=C_ENV))
        ax.text(p1[0], p1[1] + 0.12, label, ha="center", va="center", fontsize=9, color=C_ENV, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_title(title if title is not None else f"из клетки {s} просим «{ACTION_NAMES[a]}»", fontsize=10)
    return ax


def plot_q(Q, desc, ax=None, title="", vmax=None, size=3.6, fmt="{:.2f}"):
    """Q(s, a) для всех клеток: клетка разбита на четыре треугольника (←, ↓, →, ↑), лучшее действие обведено."""
    nrow, ncol = len(desc), len(desc[0])
    ax = draw_lake(desc, ax, size, letters="HG")
    Q = np.asarray(Q, float)
    vmax = vmax or max(float(Q.max()), 1e-9)
    for s in range(nrow * ncol):
        r, c = divmod(s, ncol)
        if desc[r][c] in "HG":
            continue
        cx, cy = c + 0.5, r + 0.5
        corners = {"tl": (c, r), "tr": (c + 1, r), "bl": (c, r + 1), "br": (c + 1, r + 1)}
        tris = {0: ("tl", "bl", (c + 0.2, cy)), 1: ("bl", "br", (cx, r + 0.8)),      # ←: левый, ↓: нижний
                2: ("tr", "br", (c + 0.8, cy)), 3: ("tl", "tr", (cx, r + 0.2))}       # →: правый, ↑: верхний
        best = int(Q[s].argmax())
        for a, (k1, k2, (tx, ty)) in tris.items():
            poly = Polygon([corners[k1], corners[k2], (cx, cy)], closed=True,
                           facecolor=plt.cm.YlOrRd(0.85 * max(Q[s, a], 0) / vmax), edgecolor="#bbbbbb", lw=0.4)
            ax.add_patch(poly)
            ax.text(tx, ty, fmt.format(Q[s, a]), ha="center", va="center", fontsize=6.2, color="#222",
                    weight="bold" if a == best else "normal")
        k1, k2, _ = tris[best]
        ax.add_patch(Polygon([corners[k1], corners[k2], (cx, cy)], closed=True, facecolor="none", edgecolor=C_AGENT, lw=1.8))
    ax.set_title(title, fontsize=10)
    return ax


def draw_backup(s, a, outcomes, gamma, ax=None, value_name="V", q_name="Q", title=""):
    """«Структура задачи» для одной пары (s, a): исходы среды и арифметика уравнения Беллмана.

    outcomes — список (s_next, prob, reward, value): вероятность исхода, награда за шаг и ценность того, где окажемся.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 1.1 + 1.0 * len(outcomes)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(outcomes)
    ys = np.linspace(0.86, 0.36, n) if n > 1 else [0.60]
    y0 = float(np.mean(ys))
    ax.text(0.10, y0, f"клетка {s}\nдействие {ARROWS[a]}", ha="center", va="center", fontsize=11, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc=C_AGENT, ec="none"))
    terms = []
    for y, (s_next, prob, reward, value) in zip(ys, outcomes):
        ax.add_patch(FancyArrowPatch((0.20, y0), (0.44, y), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=C_ENV,
                                     connectionstyle="arc3,rad=0"))
        ax.text(0.30, (y0 + y) / 2 + 0.03 * np.sign(y - y0 + 1e-9), f"p = {prob:.2f}".replace("0.33", "1/3"),
                fontsize=9, color=C_ENV, ha="center", va="center", weight="bold")
        ax.text(0.51, y, f"клетка {s_next}", ha="center", va="center", fontsize=10.5, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc=C_ENV, ec="none"))
        ax.text(0.65, y, f"награда за шаг  r = {reward:g}\n{value_name}(клетка {s_next}) = {value:.2f}",
                ha="left", va="center", fontsize=9.5, color="#222")
        terms.append(f"{prob:.2f}".replace("0.33", "1/3") + f"·[{reward:g} + γ·{value:.2f}]")
    total = sum(p * (r + gamma * v) for _, p, r, v in outcomes)
    ax.text(0.5, 0.09, f"{q_name}({s}, {ARROWS[a]}) = " + " + ".join(terms[:2]) + "\n+ " + " + ".join(terms[2:]) + f" = {total:.3f}",
            ha="center", va="center", fontsize=10, color=C_ACCENT, weight="bold", linespacing=1.5)
    ax.set_title(title, fontsize=10.5)
    return ax


def plot_q_updates(histories, ax=None, title="", labels=None):
    """Как меняется одно число Q(s, a) по шагам обновлений (для примера с лотереей)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, h in enumerate(histories):
        ax.plot(h, lw=1.4, label=labels[i] if labels else None)
    ax.axhline(0, color=C_GREY, lw=1, ls="--")
    ax.set_xlabel("номер обновления"); ax.set_ylabel("оценка Q(s, купить)"); ax.set_title(title, fontsize=10)
    if labels:
        ax.legend(fontsize=8)
    return ax
