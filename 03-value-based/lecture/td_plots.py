"""Картинки для лекции 3: карта Cliff Walking, жадный путь по таблице Q, кривые обучения.

Ничего содержательного здесь нет — только matplotlib. Читать не нужно.
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from matplotlib.patches import Rectangle

C_AGENT, C_ENV, C_REWARD, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#888888"
CURVE_COLORS = {"q-learning": C_AGENT, "sarsa": C_REWARD, "expected sarsa": C_ENV}

CLIFF_ROWS, CLIFF_COLS = 4, 12
CLIFF_START, CLIFF_GOAL = 36, 47
CLIFF = set(range(37, 47))                          # нижний ряд между стартом и выходом


def draw_cliff(paths=None, ax=None, title=""):
    """Карта Cliff Walking 4 × 12; paths — словарь {подпись: список клеток} для путей поверх карты."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(0, CLIFF_COLS); ax.set_ylim(CLIFF_ROWS, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in range(CLIFF_ROWS * CLIFF_COLS):
        r, c = divmod(s, CLIFF_COLS)
        color = "#f3a9a9" if s in CLIFF else "#a9dfa9" if s == CLIFF_GOAL else "#f2f2f2" if s == CLIFF_START else "white"
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#aaaaaa", lw=0.6))
    ax.text(0.5, 3.5, "S", ha="center", va="center", fontsize=13, weight="bold", color="#444")
    ax.text(11.5, 3.5, "G", ha="center", va="center", fontsize=13, weight="bold", color="#2f6b2f")
    ax.text(6.0, 3.5, "обрыв: −100 и обратно на старт", ha="center", va="center", fontsize=9.5, color="#9b2f2f")
    paths = paths or {}
    for i, (label, states) in enumerate(paths.items()):
        pts = np.array([divmod(int(s), CLIFF_COLS) for s in states], float) + 0.5
        shift = (i - (len(paths) - 1) / 2) * 0.14            # чтобы совпадающие куски путей не слипались
        color = CURVE_COLORS.get(label.lower(), C_GREY)
        ax.plot(pts[:, 1], pts[:, 0] + shift, "-o", ms=3.5, lw=2.6, color=color, label=f"{label}: {len(states) - 1} шагов")
    if paths:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(paths), frameon=False, fontsize=9.5)
    ax.set_title(title, fontsize=10)
    return ax


def greedy_path(Q, max_steps=60):
    """Клетки, по которым пройдёт жадная стратегия argmax Q из старта Cliff Walking (без случайных шагов)."""
    env = gym.make("CliffWalking-v1")
    s, _ = env.reset(seed=0)
    path = [s]
    for _ in range(max_steps):
        s, _, terminated, truncated, _ = env.step(int(Q[s].argmax()))
        path.append(s)
        if terminated or truncated:
            break
    return path


def smooth(x, w=10):
    """Скользящее среднее по окну w — чтобы кривая обучения не дрожала."""
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_curves(curves, ax=None, title="", w=10, ylim=(-120, 0)):
    """curves — словарь {название: массив (запуски, эпизоды)}; рисуем среднее по запускам, сглаженное по эпизодам."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, runs in curves.items():
        ax.plot(smooth(np.asarray(runs).mean(axis=0), w), lw=2, label=name, color=CURVE_COLORS.get(name.lower()))
    ax.set_xlabel("эпизод обучения"); ax.set_ylabel("return за эпизод")
    ax.set_ylim(*ylim); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(title, fontsize=10)
    return ax
