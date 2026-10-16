"""Такси: создание среды, карта с машиной и пассажиром, прогон эпизода, оценка стратегии.

Сигнатуры (все рисовальщики принимают ``ax=None`` и возвращают ``ax``):

    make_taxi()                                     -> env: Taxi-v4 (в старых версиях gymnasium — Taxi-v3)
    draw_taxi(env, s, ax=None, title="")            -> ax: карта 5 × 5, стены, остановки, такси, пассажир, цель
    play_episode(env, Q=None, eps=0.0, seed=0,
                 max_steps=200)                     -> (states, actions, rewards): Q=None — действия наугад
    average_greedy_return(env, Q, n_episodes=100,
                          seed=0)                   -> средний return жадной стратегии argmax Q

Ничего содержательного здесь нет — только matplotlib и обвязка Gymnasium. Читать не нужно.
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"
STOP_COLORS = {"R": "#C44E52", "G": "#55A868", "Y": "#CCB974", "B": "#4C72B0"}
ACTION_NAMES = ["вниз", "вверх", "вправо", "влево", "посадить", "высадить"]


def make_taxi(**kwargs):
    """Среда такси: Taxi-v4, а в старых версиях gymnasium (например, в Colab) — Taxi-v3."""
    try:
        return gym.make("Taxi-v4", **kwargs)
    except gym.error.Error:
        return gym.make("Taxi-v3", **kwargs)


def _layout(env):
    """Разбор карты env.unwrapped.desc: остановки {буква: (ряд, столбец)} и стены {(ряд, столбец) справа}."""
    desc = [row.tobytes().decode() if hasattr(row, "tobytes") else "".join(c.decode() for c in row)
            for row in env.unwrapped.desc]
    stops, walls = {}, set()
    for r in range(5):
        line = desc[r + 1]
        for c in range(5):
            ch = line[2 * c + 1]
            if ch in STOP_COLORS:
                stops[ch] = (r, c)
            if 2 * c + 2 < len(line) and line[2 * c + 2] == "|":
                walls.add((r, c))                     # стена справа от клетки (r, c)
    return stops, walls


def draw_taxi(env, s, ax=None, title=""):
    """Карта такси в состоянии s: остановки цветом, толстые линии — стены, жёлтая машина (зелёная — с пассажиром)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))
    taxi_r, taxi_c, pass_loc, dest = env.unwrapped.decode(s)
    stops, walls = _layout(env)
    locs = [tuple(p) for p in env.unwrapped.locs]     # позиции кодов 0–3 (R, G, Y, B) — из самой среды

    ax.set_xlim(-0.12, 5.12); ax.set_ylim(5.12, -0.12)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for r in range(5):
        for c in range(5):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="white", edgecolor="#cccccc", lw=0.8))
    for name, (r, c) in stops.items():
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=STOP_COLORS[name], alpha=0.25, edgecolor="none"))
        ax.text(c + 0.14, r + 0.26, name, fontsize=13, weight="bold", color=STOP_COLORS[name])
    for r, c in walls:                                # внутренние стены
        ax.plot([c + 1, c + 1], [r, r + 1], color="#333333", lw=5, solid_capstyle="butt")
    ax.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], color="#333333", lw=5)

    dr, dc = locs[dest]
    ax.plot([dc + 0.78], [dr + 0.24], marker="*", ms=17, color=C_ACCENT, zorder=4)   # звезда — куда везти
    in_taxi = pass_loc == 4
    if not in_taxi:                                   # пассажир ждёт на своей остановке
        pr, pc = locs[pass_loc]
        ax.add_patch(Circle((pc + 0.5, pr + 0.68), 0.16, facecolor="#333333", zorder=4))
    car = "#5fae6b" if in_taxi else "#e8c84d"         # как в оригинальном рендере: жёлтое пустое, зелёное с пассажиром
    ax.add_patch(FancyBboxPatch((taxi_c + 0.2, taxi_r + 0.3), 0.6, 0.4,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=car, edgecolor="#555555", lw=1.5, zorder=5))
    ax.set_title(title, fontsize=11)
    return ax


def play_episode(env, Q=None, eps=0.0, seed=0, max_steps=200):
    """Один эпизод: действия по argmax Q (с долей случайных eps); Q=None — полностью наугад."""
    rng = np.random.default_rng(seed)
    s, _ = env.reset(seed=seed)
    states, actions, rewards = [s], [], []
    for _ in range(max_steps):
        if Q is None or rng.random() < eps:
            a = int(rng.integers(env.action_space.n))
        else:
            a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        if terminated or truncated:
            break
    return states, actions, rewards


def average_greedy_return(env, Q, n_episodes=100, seed=0):
    """Средний return жадной стратегии argmax Q по n_episodes эпизодам."""
    totals = [sum(play_episode(env, Q, seed=seed + i)[2]) for i in range(n_episodes)]
    return float(np.mean(totals))
