"""Картинки Frozen Lake для лекции 1, семинара 1 и домашки блока 1.

Ничего содержательного здесь нет — только matplotlib. Читать не нужно,
но сигнатуры полезно знать: все функции принимают ``ax=None`` и возвращают ``ax``,
поэтому их можно класть в сетку ``plt.subplots`` и перерисовывать кадр за кадром.

    desc = lake_desc(env)                      # карта как ['SFFF', 'FHFH', ...]
    draw_lake(desc, ax=None, size=3.0)
    draw_paths(sessions, desc, ax=None, title="", color=C_AGENT, alpha=0.35, size=3.0)
    draw_policy(policy, desc, ax=None, title="", size=3.0, missing=None)
    plot_values(values, desc, ax=None, title="", vmax=None, size=3.0)
    plot_returns(returns, threshold=None, elite_mask=None, ax=None, title="")
    show_transitions(env, s, a, ax=None, title=None, size=3.0)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# палитра курса — та же, что в assets/make_figures.py и 02-environments/lecture/gridworld_plots.py
C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

ARROWS = "←↓→↑"                      # так пронумерованы действия в Frozen Lake: 0=←, 1=↓, 2=→, 3=↑
ACTION_NAMES = ["влево", "вниз", "вправо", "вверх"]
CELL_COLORS = {"S": "#f2f2f2", "F": "#ffffff", "H": "#a8c8ec", "G": "#a9dfa9"}


def lake_desc(env):
    """Карта озера как список строк: ['SFFF', 'FHFH', 'FFFH', 'HFFG']."""
    return ["".join(c.decode() for c in row) for row in env.unwrapped.desc]


def draw_lake(desc, ax=None, size=3.0, numbers=False):
    """Пустая карта озера. Возвращает оси, поверх которых можно рисовать маршруты и стрелки."""
    nrow, ncol = len(desc), len(desc[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(size, size * nrow / ncol))
    ax.set_xlim(0, ncol); ax.set_ylim(nrow, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for r in range(nrow):
        for c in range(ncol):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=CELL_COLORS[desc[r][c]], edgecolor="k", lw=0.6))
            if desc[r][c] in "SHG":
                ax.text(c + 0.5, r + 0.82, desc[r][c], ha="center", va="center", fontsize=8, color="#666")
            if numbers:
                ax.text(c + 0.06, r + 0.17, str(r * ncol + c), ha="left", va="center", fontsize=7, color=C_GREY)
    return ax


def draw_paths(sessions, desc, ax=None, title="", color=C_AGENT, alpha=0.35, size=3.0):
    """Маршруты эпизодов поверх карты: ломаная через центры клеток, точка — где эпизод закончился."""
    ax = draw_lake(desc, ax, size)
    ncol, rng = len(desc[0]), np.random.default_rng(0)
    for states, *_ in sessions:
        pts = np.array([divmod(int(s), ncol) for s in states], float) + 0.5 + rng.normal(0, 0.07, (len(states), 2))
        ax.plot(pts[:, 1], pts[:, 0], color=color, alpha=alpha, lw=1.5)
        ax.plot(pts[-1, 1], pts[-1, 0], "o", color=color, alpha=alpha, ms=4)
    ax.set_title(title, fontsize=10)
    return ax


def draw_policy(policy, desc, ax=None, title="", size=3.0, missing=None):
    """Стрелка — самое вероятное действие, насыщенность — его вероятность; «·» — строка почти равномерная.

    ``missing`` — множество клеток, для которых действие не задано явно; они помечаются «?»,
    чтобы незаполненная клетка не выглядела как осознанный выбор.
    """
    ax = draw_lake(desc, ax, size)
    ncol, missing = len(desc[0]), set(missing or ())
    for s, p in enumerate(policy):
        r, c = divmod(s, ncol)
        if desc[r][c] in "HG":
            continue
        if s in missing:
            ax.text(c + 0.5, r + 0.5, "?", ha="center", va="center", fontsize=17, color=C_ACCENT, weight="bold")
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
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=plt.cm.YlOrRd(0.85 * grid[r, c] / vmax), edgecolor="k", lw=0.6))
            ax.text(c + 0.5, r + 0.5, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    return ax


def plot_returns(returns, threshold=None, elite_mask=None, ax=None, title=""):
    """Каждая точка — эпизод, по вертикали его return (как на схеме метода Cross-Entropy)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.2))
    returns, x = np.asarray(returns, float), np.random.default_rng(0).uniform(0, 1, len(returns))
    if elite_mask is None:
        ax.scatter(x, returns, s=22, color=C_AGENT, alpha=0.7)
    else:
        m = np.asarray(elite_mask, bool)
        ax.scatter(x[~m], returns[~m], s=18, color="grey", alpha=0.35, label=f"остальные ({(~m).sum()})")
        ax.scatter(x[m], returns[m], s=42, color=C_ENV, alpha=0.95, edgecolors="white", label=f"элита ({m.sum()})")
        ax.legend(loc="upper right", fontsize=8)
    if threshold is not None:
        ax.axhline(threshold, ls="--", color=C_ACCENT, lw=1.5)
        ax.text(0.0, threshold + 0.02, f"порог = {threshold:.2f}", color=C_ACCENT, fontsize=8)
    ax.set_xticks([]); ax.set_ylabel("return"); ax.set_ylim(-0.05, 1.05); ax.set_title(title, fontsize=10)
    return ax


def show_transitions(env, s, a, ax=None, title=None, size=3.0):
    """Куда на самом деле ведёт действие ``a`` из клетки ``s``: стрелки во все исходы с вероятностями.

    На скользком льду просьба «вправо» выполняется лишь в трети случаев — эта картинка
    показывает ровно то, что лежит в ``env.unwrapped.P[s][a]``.
    """
    desc = lake_desc(env)
    ncol = len(desc[0])
    ax = draw_lake(desc, ax, size, numbers=True)
    r0, c0 = divmod(s, ncol)
    ax.add_patch(plt.Rectangle((c0, r0), 1, 1, facecolor="none", edgecolor=C_AGENT, lw=3.0))
    ax.text(c0 + 0.5, r0 + 0.5, ARROWS[a], ha="center", va="center", fontsize=22, color=C_AGENT, weight="bold")

    merged = {}
    for prob, s_next, _reward, _done in env.unwrapped.P[s][a]:
        merged[s_next] = merged.get(s_next, 0.0) + prob
    for s_next, prob in merged.items():
        r1, c1 = divmod(s_next, ncol)
        if s_next == s:
            ax.text(c0 + 0.5, r0 + 0.68, f"остаёмся {prob:.2f}", ha="center", va="center", fontsize=7.5,
                    color=C_ENV, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
            continue
        p0, p1 = np.array([c0 + 0.5, r0 + 0.5]), np.array([c1 + 0.5, r1 + 0.5])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        ax.add_patch(FancyArrowPatch(p0 + 0.28 * d, p1 - 0.32 * d, arrowstyle="-|>",
                                     mutation_scale=15, lw=2.2, color=C_ENV))
        ax.text(c1 + 0.5, r1 + 0.3, f"{prob:.2f}", ha="center", va="center", fontsize=8.5,
                color=C_ENV, weight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    ax.set_title(title if title is not None else f"из клетки {s} просим «{ACTION_NAMES[a]}»", fontsize=10)
    return ax
