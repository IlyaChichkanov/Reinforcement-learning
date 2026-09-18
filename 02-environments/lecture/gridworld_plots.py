"""Картинки клетчатого мира для лекции 2: карта, ценности, стратегия, переходы, траектории.

Ничего содержательного здесь нет — только matplotlib. Читать не нужно.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from gridworld import ARROWS, ACTION_NAMES, TERMINAL_REWARD

CELL_COLORS = {".": "#ffffff", "S": "#f2f2f2", "#": "#555555", "G": "#a9dfa9", "X": "#f3a9a9"}
C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"
CMAP, NORM = plt.cm.RdYlGn, Normalize(vmin=-1.0, vmax=1.0)


def draw_grid(env, ax=None, size=3.6, numbers=False, labels=True, start_label=True):
    """Пустая карта мира: стены тёмные, выход зелёный, яма розовая. numbers=True — подписать номера клеток."""
    if ax is None:
        _, ax = plt.subplots(figsize=(size, size * env.n_rows / env.n_cols))
    ax.set_xlim(0, env.n_cols); ax.set_ylim(env.n_rows, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            ch = env.layout[r][c]
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=CELL_COLORS[ch], edgecolor="k", lw=0.8))
            if labels and ch in TERMINAL_REWARD:
                ax.text(c + 0.5, r + 0.5, f"{TERMINAL_REWARD[ch]:+.0f}", ha="center", va="center",
                        fontsize=15, weight="bold", color="#2f6b2f" if ch == "G" else "#9b2f2f")
            if labels and start_label and ch == "S" and not numbers:
                ax.text(c + 0.5, r + 0.8, "старт", ha="center", va="center", fontsize=8.5, color=C_GREY)
            if numbers and ch != "#":
                ax.text(c + 0.06, r + 0.16, str(r * env.n_cols + c), ha="left", va="center", fontsize=7, color=C_GREY)
    return ax


def plot_values(values, env, ax=None, title="", size=3.6, fmt="{:+.2f}", numbers=False, text_y=0.5, fontsize=11):
    """Карта с числами ценности: зелёное — хорошо, красное — плохо."""
    ax = draw_grid(env, ax, size, numbers=numbers)
    values = np.asarray(values, float)
    for s in range(env.n_states):
        r, c = divmod(s, env.n_cols)
        if env.is_wall(s) or env.is_terminal(s):
            continue
        v = values[s]
        if np.isnan(v):
            ax.text(c + 0.5, r + text_y, "—", ha="center", va="center", fontsize=fontsize, color=C_GREY); continue
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=CMAP(NORM(v)), edgecolor="k", lw=0.8, alpha=0.85))
        ax.text(c + 0.5, r + text_y, fmt.format(v), ha="center", va="center", fontsize=fontsize, weight="bold", color="#222")
    ax.set_title(title, fontsize=10)
    return ax


def draw_policy(policy, env, ax=None, title="", size=3.6, values=None):
    """Стрелки — самое вероятное действие в каждой клетке; values — подложить тепловую карту."""
    ax = plot_values(values, env, ax, size=size, text_y=0.3, fontsize=10) if values is not None else draw_grid(env, ax, size)
    for s in range(env.n_states):
        r, c = divmod(s, env.n_cols)
        if env.is_wall(s) or env.is_terminal(s):
            continue
        p = np.asarray(policy[s], float)
        a, top2 = int(p.argmax()), np.sort(p)[-2:]
        y = r + 0.66 if values is not None else r + 0.5
        if top2[1] - top2[0] < 0.05:
            ax.text(c + 0.5, y, "·", ha="center", va="center", fontsize=18, color=C_GREY)
        else:
            ax.text(c + 0.5, y, ARROWS[a], ha="center", va="center",
                    fontsize=17 if values is not None else 22, color="#222")
    ax.set_title(title, fontsize=10)
    return ax


def plot_q(Q, env, ax=None, title="", size=4.0, fmt="{:+.2f}"):
    """Все четыре числа Q(s, a) в клетке: четыре треугольника, лучшее действие обведено."""
    ax = draw_grid(env, ax, size, start_label=False)
    Q = np.asarray(Q, float)
    for s in range(env.n_states):
        r, c = divmod(s, env.n_cols)
        if env.is_wall(s) or env.is_terminal(s):
            continue
        cx, cy = c + 0.5, r + 0.5
        corners = {"tl": (c, r), "tr": (c + 1, r), "bl": (c, r + 1), "br": (c + 1, r + 1)}
        tris = {0: ("tl", "bl", (c + 0.19, cy)), 1: ("bl", "br", (cx, r + 0.81)),
                2: ("tr", "br", (c + 0.81, cy)), 3: ("tl", "tr", (cx, r + 0.19))}
        best = int(Q[s].argmax())
        for a, (k1, k2, (tx, ty)) in tris.items():
            ax.add_patch(Polygon([corners[k1], corners[k2], (cx, cy)], closed=True,
                                 facecolor=CMAP(NORM(Q[s, a])), edgecolor="#bbbbbb", lw=0.4, alpha=0.85))
            ax.text(tx, ty, fmt.format(Q[s, a]), ha="center", va="center", fontsize=6.5, color="#222",
                    weight="bold" if a == best else "normal")
        k1, k2, _ = tris[best]
        ax.add_patch(Polygon([corners[k1], corners[k2], (cx, cy)], closed=True, facecolor="none",
                             edgecolor=C_AGENT, lw=2.0))
    ax.set_title(title, fontsize=10)
    return ax


def draw_transitions(env, s, a, ax=None, title=None, size=3.6):
    """Что бывает после одного действия: стрелки во все возможные клетки с вероятностями."""
    ax = draw_grid(env, ax, size, numbers=True)
    r0, c0 = divmod(s, env.n_cols)
    ax.add_patch(Rectangle((c0, r0), 1, 1, facecolor="none", edgecolor=C_AGENT, lw=3.5))
    ax.text(c0 + 0.5, r0 + 0.5, ARROWS[a], ha="center", va="center", fontsize=24, color=C_AGENT, weight="bold")
    for prob, s_next, reward, _ in env.transitions(s, a):
        r1, c1 = divmod(s_next, env.n_cols)
        label = f"{prob:g}"
        if s_next == s:
            ax.text(c0 + 0.5, r0 + 0.74, f"остаёмся: {label}", ha="center", va="center",
                    fontsize=8.5, color=C_ENV, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
            continue
        p0, p1 = np.array([c0 + 0.5, r0 + 0.5]), np.array([c1 + 0.5, r1 + 0.5])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        ax.add_patch(FancyArrowPatch(p0 + 0.3 * d, p1 - 0.34 * d, arrowstyle="-|>", mutation_scale=18,
                                     lw=2.6, color=C_ENV))
        ax.text(c1 + 0.5, r1 + 0.22, label, ha="center", va="center", fontsize=10,
                color=C_ENV, weight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_title(title if title is not None else f"из клетки {s} просим «{ACTION_NAMES[a]}»", fontsize=10)
    return ax


def draw_paths(sessions, env, ax=None, title="", size=3.6, color=C_AGENT, alpha=0.3):
    """Траектории сыгранных эпизодов поверх карты; точка — где эпизод закончился."""
    ax = draw_grid(env, ax, size)
    rng = np.random.default_rng(0)
    for states, *_ in sessions:
        pts = np.array([divmod(int(s), env.n_cols) for s in states], float) + 0.5 + rng.normal(0, 0.06, (len(states), 2))
        ax.plot(pts[:, 1], pts[:, 0], color=color, alpha=alpha, lw=1.4)
        ax.plot(pts[-1, 1], pts[-1, 0], "o", color=color, alpha=alpha, ms=5)
    ax.set_title(title, fontsize=10)
    return ax


def draw_backup(env, s, a, rows, gamma, ax=None, value_name="max Q*", q_name="Q*", title=""):
    """rows — список (вероятность, следующая клетка, награда за шаг, ценность следующей клетки)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.6, 1.3 + 0.95 * len(rows)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = np.linspace(0.88, 0.42, len(rows)) if len(rows) > 1 else [0.65]
    y0 = float(np.mean(ys))
    ax.text(0.09, y0, f"клетка {s}\nдействие {ARROWS[a]}", ha="center", va="center", fontsize=11.5, weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.5", fc=C_AGENT, ec="none"))
    terms = []
    for y, (prob, s_next, reward, value) in zip(ys, rows):
        ax.add_patch(FancyArrowPatch((0.19, y0), (0.42, y), arrowstyle="-|>", mutation_scale=16, lw=1.9, color=C_ENV))
        ax.text(0.30, (y0 + y) / 2 + 0.04 * np.sign(y - y0 + 1e-9), f"p = {prob:g}", fontsize=9.5,
                color=C_ENV, ha="center", va="center", weight="bold")
        ax.text(0.49, y, f"клетка {s_next}", ha="center", va="center", fontsize=10.5, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc=C_ENV, ec="none"))
        ax.text(0.60, y, f"награда за шаг  r = {reward:+.2f}\n{value_name}(клетка {s_next}) = {value:+.2f}",
                ha="left", va="center", fontsize=9.5, color="#222")
        terms.append(f"{prob:g}·[{reward:+.2f} + γ·{value:+.2f}]")
    total = sum(p * (r + gamma * v) for p, _, r, v in rows)
    half = (len(terms) + 1) // 2
    text = " + ".join(terms[:half]) + ("\n+ " + " + ".join(terms[half:]) if terms[half:] else "")
    ax.text(0.5, 0.14, f"{q_name}({s}, {ARROWS[a]}) = " + text + f" = {total:+.3f}", ha="center", va="center",
            fontsize=10.5, color=C_ACCENT, weight="bold", linespacing=1.6)
    ax.set_title(title, fontsize=10.5)
    return ax


def plot_q_updates(histories, ax=None, title="", labels=None, ylabel="оценка Q(s, купить)"):
    """Как меняется одно число Q(s, a) по шагам обновлений (пример с лотереей)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, h in enumerate(histories):
        ax.plot(h, lw=1.3, label=labels[i] if labels else None)
    ax.axhline(0, color=C_GREY, lw=1, ls="--")
    ax.set_xlabel("номер обновления"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    if labels:
        ax.legend(fontsize=8)
    return ax
