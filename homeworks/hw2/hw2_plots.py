"""Обвязка для домашки блока 2: озеро со случайным стартом, модель среды из Gymnasium, лог такси, горная машина на сетке.

Ничего содержательного здесь нет — только numpy, matplotlib и обвязка Gymnasium.
Читать не нужно, сигнатуры полезно знать (все рисовальщики принимают ``ax=None`` и возвращают ``ax``):

    make_lake(random_start=False)                 # FrozenLake 8×8 без скольжения; random_start — эпизод из случайной клетки
    P, R, terminal = mdp_from_gym(env)            # модель P[s, a, s'], R[s, a] из env.unwrapped.P
    success_rate_greedy(env, Q, n_episodes=200, seed=0)   # доля эпизодов, где argmax Q дошёл до награды
    log = collect_random_log(env, n_episodes=3000, seed=0, max_steps=200)   # пятёрки (s, a, r, s', a', done)
    animate_taxi(env, states, rewards=None, fps=3)  # GIF поездки такси по списку состояний (показать: display(...))
    plot_learning_curves(curves, ax=None, title="", w=50, ylabel=None)  # {имя: массив по эпизодам} сглаженно
    plot_vs(x, curves, xlabel, ylabel, ax=None, title="", logx=False)  # {имя: значения} против x — метрика против параметра
    make_mountain_car()                           # MountainCar-v0 с лимитом 200 шагов
    discretize = make_discretize(env, bins)       # функция: наблюдение (позиция, скорость) -> пара номеров клеток сетки
    greedy_episode(env, Q, discretize, seed=0, max_steps=200)   # -> (positions, reached)
    plot_value_grid(Q, env, ax=None, title="")    # тепловая карта max_a Q по сетке позиция × скорость
    plot_positions(runs, ax=None, title="")       # {подпись: positions} — позиция машины по шагам
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"


# ---------- озеро 8×8 ----------

class RandomStart(gym.Wrapper):
    """Каждый эпизод начинается в случайной нетерминальной клетке, а не в углу."""

    def __init__(self, env):
        super().__init__(env)
        desc = env.unwrapped.desc
        flat = "".join(row.tobytes().decode() if hasattr(row, "tobytes") else "".join(c.decode() for c in row)
                       for row in desc)
        self.start_cells = np.array([i for i, ch in enumerate(flat) if ch in "SF"])
        self._rng = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        s = int(self._rng.choice(self.start_cells))
        self.env.unwrapped.s = s
        return s, info


def make_lake(random_start=False):
    """Озеро 8×8 без скольжения (карта домашки 1); random_start=True — старт в случайной клетке льда."""
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    return RandomStart(env) if random_start else env


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
    """Доля эпизодов, в которых стратегия «действие с наибольшим Q» получила положительную награду."""
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


def plot_learning_curves(curves, ax=None, title="", w=50, ylabel=None):
    """Сглаженные скользящим окном w кривые по эпизодам: {имя: массив}; ylabel — подпись оси (по умолчанию return)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        smooth = np.convolve(values, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(smooth)) + w, smooth, lw=2, label=name)
    ax.set_xlabel("эпизод", fontsize=12)
    ax.set_ylabel(ylabel or f"return при обучении (среднее по {w})", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    return ax


def plot_vs(x, curves, xlabel, ylabel, ax=None, title="", logx=False):
    """Метрика против параметра: {имя: список значений той же длины, что x}, точки соединены линиями."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    for name, values in curves.items():
        ax.plot(x, values, lw=2, marker="o", ms=6, label=name)
    if logx:
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    return ax


# ---------- анимация такси ----------

def animate_taxi(env, states, rewards=None, fps=3, path="taxi_greedy.gif"):
    """GIF проезда такси по списку состояний (кадр на шаг); возвращает IPython Image — показать через display()."""
    import io
    from PIL import Image as PILImage
    from IPython.display import Image
    from taxi_plots import draw_taxi

    frames, total = [], 0.0
    for k, s in enumerate(states):
        if rewards is not None and k > 0:
            total += rewards[k - 1]
        title = f"шаг {k}" + (f", сумма наград {total:+.0f}" if rewards is not None else "")
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        draw_taxi(env, s, ax=ax, title=title)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72)
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).convert("P", palette=PILImage.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    with open(path, "rb") as f:
        return Image(data=f.read(), format="gif")


# ---------- горная машина на сетке ----------

def make_mountain_car():
    """MountainCar-v0: наблюдение (позиция, скорость), три действия, −1 за шаг, лимит 200 шагов."""
    return gym.make("MountainCar-v0")


def make_discretize(env, bins):
    """Сетка bins × bins по границам среды: возвращает функцию obs -> (номер клетки позиции, номер клетки скорости)."""
    low, high = env.observation_space.low, env.observation_space.high

    def discretize(obs):
        idx = ((np.asarray(obs) - low) / (high - low) * bins).astype(int)
        return tuple(int(i) for i in np.clip(idx, 0, bins - 1))
    return discretize


def greedy_episode(env, Q, discretize, seed=0, max_steps=200):
    """Один эпизод по таблице Q на сетке (действие с наибольшим Q, без случайных шагов); возвращает позиции и флаг «доехала»."""
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
