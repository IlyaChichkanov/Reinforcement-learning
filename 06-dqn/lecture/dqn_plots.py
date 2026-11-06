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
