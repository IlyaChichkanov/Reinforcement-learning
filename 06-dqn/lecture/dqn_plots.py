"""Служебный код и картинки лекции 6: таблица на сетке для CartPole, след одного обновления, кривые обучения.

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.
Тележку рисуют draw_cartpole и animate_cartpole из модуля семинара 5 (cartpole_plots.py).

Таблица (Q-learning лекций 3–4 на сетке)
    LOW, HIGH                     границы сетки: x ±2.4, ẋ ±3, φ ±0.21 рад, φ̇ ±3.5 рад/с
    GridTable(bins=6)             сетка bins⁴ клеток; .cell(s) — номер клетки, .n — число клеток
    train_table(table, seed=0, episodes=400, alpha=0.1, gamma=0.99, eps_steps=10_000)
                                  Q-learning на сетке; ε падает от 1 до 0.05 за eps_steps шагов;
                                  возвращает (Q, длины эпизодов, сколько раз обновлялась каждая клетка)

След одного обновления на плоскости (угол φ, угловая скорость φ̇), тележка в центре и стоит (x = ẋ = 0)
    plane(k=101)                  точки плоскости: массив (k·k, 4) и форма (k, k)
    footprint_table(table, point, points)        1 там, где точка в той же клетке, что point, иначе 0
    footprint_net(qnet, point, points, action=1, target=50.0, lr=0.01)
                                  один шаг градиента ½(target − Q(point, action))² по весам сети;
                                  возвращает (сдвиг Q(·, action) во всех точках, сдвиг в самой point)

Рисовальщики (принимают ax=None, рисуют, возвращают ax)
    plot_footprint(ratio, shape, point, ax=None, title="")      1 — сдвиг как в point, 0 — не сдвинулась
    plot_runs(curves, ax=None, title="", window=10)
        curves — словарь {подпись: массив (запуски, эпизоды) длин}; тонкие линии — запуски, жирная — медиана
    plot_q_start(curves, ax=None, title="")
        curves — словарь {подпись: массив (запуски, эпизоды) оценок Q старта}; шкала логарифмическая,
        пунктир — наибольшая возможная ценность 1 / (1 − γ) = 100
    plot_w_curves(curves, ax=None, title="")    пример «w и 2w»: оценка v̂(1) по шагам, шкала логарифмическая

Картинки таблицы и сети на плоскости (φ, φ̇)
    plane_cells(table, states)                  номера клеток (по φ, по φ̇) для списка состояний
    plot_grid_episode(table, states, ax=None, title="")   сетка таблицы и точки одного эпизода; клетки, где побывал агент, закрашены
    table_slice(Q, table, visits, x=0, x_dot=0) -> (V, N)  срез таблицы: V = max Q и число обновлений по клеткам (φ, φ̇)
    neighbour_pair(V, N)                        пустая клетка и её самая обновляемая соседка — ((i, j), (i, j)) или None
    plot_table_values(V, N, table, ax=None, title="", pair=None, vmax=None)   карта V среза; белые клетки не обновлялись; пара — ☆ и ★
    q_plane(qnet, k=101)                        V = max Q сети по точкам плоскости — массив k × k
    plot_q_plane(V, ax=None, title="", vmax=None)
    plot_batch_sources(sources, episode, ax=None, title="")   из каких эпизодов пришли переходы пачки: {подпись: номера эпизодов}
    plot_grid_2x2(lengths, q_max, window=10)    четыре панели «буфер да/нет × целевая сеть да/нет»; lengths и q_max — словари по названиям вариантов
    plot_q_lag(trace, ax=None, title="", window=None)   оценка старта основной и целевой сети по шагам; trace — (шаги, 2)
"""
import numpy as np
import matplotlib.pyplot as plt

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

LOW = np.array([-2.4, -3.0, -0.21, -3.5])
HIGH = -LOW
MAX_STEPS = 500

# один цвет у варианта на всех картинках лекции
COLORS = {"таблица 6⁴": C_GREY, "по шагу подряд": C_REWARD, "буфер": C_ACCENT,
          "целевая сеть": C_ENV, "DQN": C_AGENT, "буфер + целевая сеть (DQN)": C_AGENT}
_SPARE = ["#8172B3", "#937860", "#DA8BC3"]


def _color(name, k):
    return COLORS.get(name, _SPARE[k % len(_SPARE)])


def make_cartpole():
    import gymnasium as gym
    return gym.make("CartPole-v1")


# ---------------------------------------------------------------------------------------------- таблица
class GridTable:
    """Сетка bins × bins × bins × bins по (x, ẋ, φ, φ̇); номер клетки — одно число."""

    def __init__(self, bins=6):
        self.bins = bins
        self.n = bins ** 4

    def cell(self, s):
        idx = ((np.asarray(s, dtype=float) - LOW) / (HIGH - LOW) * self.bins).astype(int)
        idx = np.clip(idx, 0, self.bins - 1)
        return int(np.ravel_multi_index(tuple(idx), (self.bins,) * 4))


def train_table(table, seed=0, episodes=400, alpha=0.1, gamma=0.99, eps_steps=10_000):
    """Q-learning на сетке; возвращает (Q, длины эпизодов, сколько раз обновлялась каждая клетка)."""
    rng, env = np.random.default_rng(seed), make_cartpole()
    Q, visits, lengths, step = np.zeros((table.n, 2)), np.zeros(table.n, dtype=int), [], 0
    for episode in range(episodes):
        s, _ = env.reset(seed=seed * 10_000 + episode)
        c = table.cell(s)
        for t in range(1, MAX_STEPS + 1):
            eps = max(0.05, 1 - step / eps_steps)
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[c]))
            s, r, terminated, truncated, _ = env.step(a)
            c2 = table.cell(s)
            target = r if terminated else r + gamma * Q[c2].max()
            Q[c, a] += alpha * (target - Q[c, a])
            visits[c] += 1; step += 1; c = c2
            if terminated or truncated:
                break
        lengths.append(t)
    return Q, np.array(lengths), visits


# ---------------------------------------------------------------------------------------------- след обновления
PHI_LIMIT, PHI_DOT_LIMIT = 0.21, 2.0


def plane(k=101):
    """Точки плоскости (φ, φ̇) при x = ẋ = 0: массив (k·k, 4) и форма (k, k); строки — φ̇, столбцы — φ."""
    P, D = np.meshgrid(np.linspace(-PHI_LIMIT, PHI_LIMIT, k), np.linspace(-PHI_DOT_LIMIT, PHI_DOT_LIMIT, k))
    points = np.zeros((k * k, 4), dtype=np.float32)
    points[:, 2], points[:, 3] = P.ravel(), D.ravel()
    return points, P.shape


def footprint_table(table, point, points):
    """У таблицы шаг меняет одну клетку: 1 там, где точка лежит в клетке point, иначе 0."""
    home = table.cell(point)
    return np.array([float(table.cell(p) == home) for p in points])


def footprint_net(qnet, point, points, action=1, target=50.0, lr=0.01):
    """Один шаг градиента ½(target − Q(point, action))²; сдвиг Q(·, action) во всех точках и в самой point."""
    import torch
    inputs = torch.as_tensor(np.asarray(points, dtype=np.float32))
    x0 = torch.as_tensor(np.asarray(point, dtype=np.float32)[None])
    with torch.no_grad():
        before, before0 = qnet(inputs)[:, action], qnet(x0)[0, action].item()
    optimizer = torch.optim.SGD(qnet.parameters(), lr=lr)
    loss = 0.5 * (target - qnet(x0)[0, action]) ** 2
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        after, after0 = qnet(inputs)[:, action], qnet(x0)[0, action].item()
    return (after - before).numpy(), after0 - before0


# ---------------------------------------------------------------------------------------------- картинки
def _plane_axes(ax):
    ax.set_xlabel("угол шеста φ, рад (плюс — наклон вправо)", fontsize=12)
    ax.set_ylabel("угловая скорость φ̇, рад/с", fontsize=12)
    ax.tick_params(labelsize=11)


def plot_footprint(ratio, shape, point, ax=None, title=""):
    """Карта следа: во сколько раз сдвинулась Q в каждой точке по сравнению со сдвигом в самой точке (★)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(np.asarray(ratio).reshape(shape), origin="lower", aspect="auto", cmap="coolwarm",
                   vmin=-1.2, vmax=1.2, extent=(-PHI_LIMIT, PHI_LIMIT, -PHI_DOT_LIMIT, PHI_DOT_LIMIT))
    ax.plot(point[2], point[3], marker="*", ms=16, color="k")
    _plane_axes(ax)
    plt.colorbar(im, ax=ax).set_label("сдвиг / сдвиг в ★", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_runs(curves, ax=None, title="", window=10):
    """Длины эпизодов нескольких вариантов: тонкие линии — запуски, жирная — медиана по запускам."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    kernel = np.ones(window) / window
    for k, (name, runs) in enumerate(curves.items()):
        color = _color(name, k)
        runs = np.atleast_2d(np.asarray(runs, dtype=float))
        smooth = np.array([np.convolve(r, kernel, mode="valid") for r in runs])
        episodes = np.arange(window, runs.shape[1] + 1)
        for run in smooth:
            ax.plot(episodes, run, color=color, lw=1, alpha=0.3)
        ax.plot(episodes, np.median(smooth, axis=0), color=color, lw=3, label=name)
    ax.axhline(MAX_STEPS, color="k", ls="--", lw=1, zorder=0, label="максимум 500")
    ax.set_ylim(0, 520); ax.grid(alpha=0.3)
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("сколько шагов простоял шест", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_title(title, fontsize=12)
    return ax


def plot_q_start(curves, ax=None, title="", gamma=0.99):
    """Оценка Q в стартовом состоянии по эпизодам; выше 1 / (1 − γ) настоящая ценность быть не может."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for k, (name, runs) in enumerate(curves.items()):
        color = _color(name, k)
        runs = np.atleast_2d(np.abs(np.asarray(runs, dtype=float)))
        for run in runs:
            ax.plot(run, color=color, lw=1, alpha=0.3)
        ax.plot(np.median(runs, axis=0), color=color, lw=3, label=name)
    ax.axhline(1 / (1 - gamma), color="k", ls="--", lw=1.2, label="предел 1 + 0.99 + 0.99² + … = 100")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="major")
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("max Q в стартовом состоянии", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.set_title(title, fontsize=12)
    return ax


def plot_w_curves(curves, ax=None, title=""):
    """Оценка v̂(1) по шагам обучения для нескольких вариантов; логарифмическая шкала."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = [C_ACCENT, C_AGENT, C_ENV, C_REWARD]
    for (name, values), color in zip(curves.items(), colors):
        ax.plot(np.abs(values), color=color, lw=2.5, label=name)
    ax.set_yscale("log")
    ax.axhline(1, color=C_GREY, lw=1, ls=":")
    ax.set_xlabel("шаг обучения", fontsize=12); ax.set_ylabel("оценка v̂(1)", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3, which="major")
    ax.legend(loc="lower left", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


# ---------------------------------------------------------------------------------------------- картинки §1–§5 (ревизия 06.11)
def _grid_lines(ax, table):
    """Линии сетки таблицы по углу и угловой скорости на плоскости (φ, φ̇)."""
    for k in range(1, table.bins):
        ax.axvline(LOW[2] + k * (HIGH[2] - LOW[2]) / table.bins, color=C_GREY, lw=0.8)
        ax.axhline(LOW[3] + k * (HIGH[3] - LOW[3]) / table.bins, color=C_GREY, lw=0.8)


def plane_cells(table, states):
    """Номера клеток (по φ, по φ̇) для списка состояний — проекция сетки на плоскость (φ, φ̇)."""
    idx = ((np.asarray(states, dtype=float)[:, 2:] - LOW[2:]) / (HIGH[2:] - LOW[2:]) * table.bins).astype(int)
    return np.clip(idx, 0, table.bins - 1)


def plot_grid_episode(table, states, ax=None, title=""):
    """Состояния одного эпизода точками на плоскости (φ, φ̇), поверх — сетка таблицы; клетки, где побывал агент, закрашены."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    states = np.asarray(states, dtype=float)
    cells = {tuple(c) for c in plane_cells(table, states)}
    w = (HIGH[2:] - LOW[2:]) / table.bins
    for i, j in cells:
        ax.add_patch(plt.Rectangle((LOW[2] + i * w[0], LOW[3] + j * w[1]), w[0], w[1], facecolor=C_AGENT, alpha=0.18, edgecolor="none"))
    _grid_lines(ax, table)
    ax.plot(states[:, 2], states[:, 3], "-", color=C_GREY, lw=0.8, alpha=0.7)
    ax.scatter(states[:, 2], states[:, 3], s=18, color=C_AGENT, zorder=3, label=f"{len(states)} состояний эпизода")
    ax.plot(states[0, 2], states[0, 3], "o", ms=9, color=C_ENV, zorder=4, label="старт")
    ax.set_xlim(LOW[2], HIGH[2]); ax.set_ylim(LOW[3], HIGH[3])
    _plane_axes(ax)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_title(title or f"эпизод прошёл через {len(cells)} клеток из {table.bins ** 2} на этой плоскости", fontsize=12)
    return ax


def table_slice(Q, table, visits, x=0.0, x_dot=0.0):
    """Срез таблицы при заданных x, ẋ: V = max_a Q и число обновлений по клеткам (φ, φ̇) — массивы bins × bins."""
    base = np.array([x, x_dot, 0.0, 0.0])
    i0 = ((base[:2] - LOW[:2]) / (HIGH[:2] - LOW[:2]) * table.bins).astype(int).clip(0, table.bins - 1)
    V, N = np.zeros((table.bins, table.bins)), np.zeros((table.bins, table.bins), dtype=int)
    for i in range(table.bins):
        for j in range(table.bins):
            c = int(np.ravel_multi_index((i0[0], i0[1], i, j), (table.bins,) * 4))
            V[i, j], N[i, j] = Q[c].max(), visits[c]
    return V, N


def neighbour_pair(V, N):
    """Клетка без обновлений, у которой соседка (по φ или φ̇) обновлялась чаще всех: ((i, j) пустая, (i, j) соседка)."""
    best = None
    for i, j in zip(*np.nonzero(N == 0)):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < N.shape[0] and 0 <= b < N.shape[1] and N[a, b] > 0 and (best is None or N[a, b] > N[best[1]]):
                best = ((i, j), (a, b))
    return best


def plot_table_values(V, N, table, ax=None, title="", pair=None, vmax=None):
    """Карта V = max Q по клеткам среза таблицы: белые клетки не обновлялись ни разу; пара клеток помечена ★ и ☆."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    shown = np.ma.masked_where(N.T == 0, V.T)                       # строки — φ̇, столбцы — φ
    im = ax.imshow(shown, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                   extent=(LOW[2], HIGH[2], LOW[3], HIGH[3]))
    ax.set_facecolor("white")
    _grid_lines(ax, table)
    w = (HIGH[2:] - LOW[2:]) / table.bins
    if pair is not None:
        for (i, j), marker, color in zip(pair, ("☆", "★"), ("k", "k")):
            ax.text(LOW[2] + (i + 0.5) * w[0], LOW[3] + (j + 0.5) * w[1], marker, ha="center", va="center",
                    fontsize=20, color=color)
    _plane_axes(ax)
    plt.colorbar(im, ax=ax).set_label("V = max Q (белые — ни одного обновления)", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def q_plane(qnet, k=101):
    """V = max_a Q_θ по точкам плоскости (φ, φ̇) при x = ẋ = 0 — массив k × k."""
    import torch
    points, shape = plane(k)
    with torch.no_grad():
        return qnet(torch.as_tensor(points)).max(dim=1).values.numpy().reshape(shape)


def plot_q_plane(V, ax=None, title="", vmax=None):
    """Карта V = max Q сети на плоскости (φ, φ̇): оценка есть в каждой точке."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(V, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                   extent=(LOW[2], HIGH[2], LOW[3], HIGH[3]))
    _plane_axes(ax)
    plt.colorbar(im, ax=ax).set_label("V = max Q", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_batch_sources(sources, episode, ax=None, title=""):
    """Из каких эпизодов пришли переходы одной пачки: гистограмма номеров эпизодов для каждого способа."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.arange(0, episode + 2, max(1, (episode + 1) // 30))
    for k, (name, eps) in enumerate(sources.items()):
        ax.hist(eps, bins=bins, color=_color(name, k), alpha=0.75, label=name)
    ax.axvline(episode, color="k", ls="--", lw=1)
    ax.text(episode, ax.get_ylim()[1] * 0.95, f"текущий эпизод {episode}", ha="right", va="top", fontsize=11)
    ax.set_xlabel("из какого эпизода переход", fontsize=12); ax.set_ylabel("переходов в пачке", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_grid_2x2(lengths, q_max, window=10):
    """Четыре варианта «буфер да/нет × целевая сеть да/нет»: в каждой панели кривые длин и наибольшая оценка старта."""
    names = [["по шагу подряд", "целевая сеть"], ["буфер", "буфер + целевая сеть (DQN)"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for r in range(2):
        for c in range(2):
            name, ax = names[r][c], axes[r, c]
            plot_runs({name: lengths[name]}, ax=ax, window=window)
            ax.legend_.remove()
            q = ", ".join(str(int(round(v))) for v in q_max[name])
            ax.text(0.03, 0.95, f"max Q старта: {q}", transform=ax.transAxes, ha="left", va="top", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor=_color(name, 0), lw=2))
            ax.set_title(name, fontsize=13, color=_color(name, 0), weight="bold")
            if r == 0:
                ax.set_xlabel("")
            if c == 1:
                ax.set_ylabel("")
    fig.tight_layout()
    return axes


def plot_q_lag(trace, ax=None, title="", window=None):
    """Оценка стартового состояния по шагам обучения: основная сеть и её копия (trace — массив (шаги, 2)).

    window — (от, до): показать только этот отрезок шагов, чтобы разглядеть, как копия отстаёт."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.8))
    trace = np.asarray(trace, dtype=float)
    lo, hi = window if window else (0, len(trace))
    steps = np.arange(lo, hi)
    ax.plot(steps, trace[lo:hi, 0], color=C_AGENT, lw=1.2, label="основная сеть — её мы учим")
    ax.plot(steps, trace[lo:hi, 1], color=C_ACCENT, lw=2.5, label="целевая сеть — копия, считает цель y")
    ax.set_xlabel("шаг обучения", fontsize=12); ax.set_ylabel("max Q в стартовом состоянии", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax
