"""Обвязка для домашки блока 2: модель среды из Gymnasium, лог такси, горная машина на сетке.

Ничего содержательного здесь нет — только numpy, matplotlib и обвязка Gymnasium.
Читать не нужно, сигнатуры полезно знать (все рисовальщики принимают ``ax=None`` и возвращают ``ax``):

    P, R, terminal = mdp_from_gym(env)            # модель P[s, a, s'], R[s, a] из env.unwrapped.P
    success_rate_greedy(env, Q, n_episodes=200, seed=0)   # доля эпизодов, где argmax Q дошёл до награды
    greedy_path(env, Q, max_steps=100)            # состояния жадного пути из старта (среда без случайности)
    log = collect_random_log(env, n_episodes=3000, seed=0, max_steps=200)   # пятёрки (s, a, r, s', a', done)
    plot_learning_curves(curves, ax=None, title="", w=50)  # {имя: массив return'ов} сглаженно
    make_mountain_car()                           # MountainCar-v0 с лимитом 200 шагов
    greedy_episode(env, Q, discretize, seed=0, max_steps=200)   # -> (positions, reached)
    plot_value_grid(Q, env, ax=None, title="")    # тепловая карта max_a Q по сетке позиция × скорость
    plot_positions(runs, ax=None, title="")       # {подпись: positions} — позиция машины по шагам
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"


# ---------- модель табличной среды Gymnasium ----------

def mdp_from_gym(env):
    """P[s, a, s'], R[s, a] и маска терминальных состояний из словаря env.unwrapped.P.

    Награда R[s, a] — ожидаемая по исходам; терминальные состояния в словаре ведут сами в себя
    с нулевой наградой, поэтому value iteration из блока 1 работает без изменений.
    """
    table = env.unwrapped.P
    n_states, n_actions = env.observation_space.n, env.action_space.n
    P, R = np.zeros((n_states, n_actions, n_states)), np.zeros((n_states, n_actions))
    terminal = np.zeros(n_states, dtype=bool)
    for s in table:
        for a in table[s]:
            for prob, s_next, reward, done in table[s][a]:
                P[s, a, s_next] += prob
                R[s, a] += prob * reward
                if done:
                    terminal[s_next] = True
    return P, R, terminal


def success_rate_greedy(env, Q, n_episodes=200, seed=0, max_steps=100):
    """Доля эпизодов, в которых жадная стратегия argmax Q получила положительную награду."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        total = 0.0
        for _ in range(max_steps):
            s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
            total += r
            if terminated or truncated:
                break
        hits += total > 0
    return hits / n_episodes


def greedy_path(env, Q, max_steps=100):
    """Состояния жадного пути argmax Q из стартового состояния; для сред без случайности."""
    s, _ = env.reset(seed=0)
    path = [s]
    for _ in range(max_steps):
        s, _, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        path.append(s)
        if terminated or truncated:
            break
    return path


# ---------- лог случайной стратегии ----------

def collect_random_log(env, n_episodes=3000, seed=0, max_steps=200):
    """Случайная стратегия играет n_episodes эпизодов; каждый переход записан пятёркой (s, a, r, s', a', done)."""
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    log = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        a = int(rng.integers(n_actions))
        for _ in range(max_steps):
            s_next, r, terminated, truncated, _ = env.step(a)
            a_next = int(rng.integers(n_actions))
            log.append((s, a, r, s_next, a_next, terminated))
            s, a = s_next, a_next
            if terminated or truncated:
                break
    return log


def plot_learning_curves(curves, ax=None, title="", w=50):
    """Сглаженные скользящим окном w кривые return'ов: {имя: массив}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        smooth = np.convolve(values, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(smooth)) + w, smooth, lw=2, label=name)
    ax.set_xlabel("эпизод", fontsize=12)
    ax.set_ylabel(f"return при обучении (среднее по {w})", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    return ax


# ---------- горная машина на сетке ----------

def make_mountain_car():
    """MountainCar-v0: наблюдение (позиция, скорость), три действия, −1 за шаг, лимит 200 шагов."""
    return gym.make("MountainCar-v0")


def greedy_episode(env, Q, discretize, seed=0, max_steps=200):
    """Один эпизод жадной стратегией по таблице Q на сетке; возвращает позиции по шагам и флаг «доехала»."""
    obs, _ = env.reset(seed=seed)
    positions, reached = [float(obs[0])], False
    for _ in range(max_steps):
        s = discretize(obs)
        obs, _, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        positions.append(float(obs[0]))
        if terminated:
            reached = True
            break
        if truncated:
            break
    return positions, reached


def plot_value_grid(Q, env, ax=None, title=""):
    """Тепловая карта V = max_a Q по сетке: по горизонтали позиция, по вертикали скорость."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))
    low, high = env.observation_space.low, env.observation_space.high
    V = Q.max(axis=-1).T                                  # строки — скорость, столбцы — позиция
    im = ax.imshow(V, origin="lower", aspect="auto", cmap="viridis",
                   extent=[low[0], high[0], low[1], high[1]])
    ax.axvline(0.5, color=C_ACCENT, lw=2, ls="--")
    ax.text(0.5, high[1] * 0.92, "флаг", color=C_ACCENT, fontsize=12, ha="right")
    ax.set_xlabel("позиция", fontsize=12)
    ax.set_ylabel("скорость", fontsize=12)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="V = max_a Q")
    return ax


def plot_positions(runs, ax=None, title=""):
    """Позиция машины по шагам для нескольких эпизодов: {подпись: positions}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    for name, positions in runs.items():
        ax.plot(positions, lw=2, label=name)
    ax.axhline(0.5, color=C_ACCENT, lw=2, ls="--", label="флаг (позиция 0.5)")
    ax.axhline(-0.5, color=C_GREY, lw=1, ls=":", label="дно долины")
    ax.set_xlabel("шаг", fontsize=12)
    ax.set_ylabel("позиция", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    return ax
