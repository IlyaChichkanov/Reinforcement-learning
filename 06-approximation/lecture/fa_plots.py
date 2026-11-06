"""Среда и картинки лекции 6: горная машина, признаки (сетка и плитки), след одного обновления, пример «w и 2w».

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.

Среда
    make_mountain_car()           MountainCar-v0: наблюдение (позиция x, скорость v), действия 0 — толкать ←,
                                  1 — не толкать, 2 — толкать →; −1 за шаг; конец у флага (x ≥ 0.5) или через 200 шагов
    LOW, HIGH                     границы наблюдения: x ∈ [−1.2, 0.6], v ∈ [−0.07, 0.07]
    normalize(s)                  наблюдение -> два числа в [−1, 1] (так их подают на вход сети)

Признаки: у каждого набора .n — число признаков, вызов f(s) -> номера признаков, равных 1 (остальные равны 0)
    GridFeatures(bins=20)         сетка bins × bins: ровно одна единица — номер клетки; q̂ на таких признаках — таблица
    TileFeatures(n=8, tilings=8)  tilings сеток n × n, каждая сдвинута на (1, 3)·t / tilings ширины плитки:
                                  по одной единице в каждой сетке, всего tilings единиц

След одного обновления (во сколько раз оценка в точке s′ сдвинулась по сравнению со сдвигом в точке s)
    plane(k=121)                  сетка точек (x, v) для карт: массив (k·k, 2) и форма (k, k)
    footprint_linear(features, point, points)   для линейных признаков: число общих единиц f(s′) и f(s), делённое
                                                на число единиц f(s)
    footprint_net(net, point, points, target, lr) один шаг градиента сети в точке point к цели target;
                                                возвращает (сдвиг во всех точках points, сдвиг в point)

Рисовальщики (принимают ax=None, рисуют, возвращают ax)
    draw_mountain_car(x=None, ax=None, title="")      горка, флаг и машина в позиции x
    plot_visits(visits, ax=None, title="")            visits — массив (bins, bins): сколько раз обновлялась клетка
    draw_tilings(features, point, ax=None, title="", show=3, around=2.5)
                                  первые show сеток плиток и плитки, где лежит point (фрагмент поля вокруг point)
    plot_footprint(ratio, shape, point, ax=None, title="")      карта следа: 1 — как в самой точке, 0 — не сдвинулась
    plot_lengths(curves, ax=None, title="", window=10)
        curves — словарь {подпись: массив (запуски, эпизоды)}; тонкие линии — запуски, жирная — медиана
    plot_w_curves(curves, ax=None, title="")
        curves — словарь {подпись: оценки V(1) по обновлениям}; шкала логарифмическая
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

LOW = np.array([-1.2, -0.07])
HIGH = np.array([0.6, 0.07])
FLAG = 0.5
MAX_STEPS = 200


def make_mountain_car():
    """MountainCar-v0 из Gymnasium (лимит 200 шагов уже встроен)."""
    import gymnasium as gym
    return gym.make("MountainCar-v0")


def normalize(s):
    """Позиция и скорость -> числа в [−1, 1]: у скорости диапазон в 13 раз уже, без этого сеть её почти не замечает."""
    s = np.asarray(s, dtype=float)
    return 2 * (s - LOW) / (HIGH - LOW) - 1


# ---------------------------------------------------------------------------------------------- признаки
class GridFeatures:
    """Сетка bins × bins по позиции и скорости: признак — «точка лежит в клетке i», единица ровно одна."""

    def __init__(self, bins=20):
        self.bins = bins
        self.n = bins * bins

    def cell(self, s):
        """Номера отрезков позиции и скорости, в которые попала точка s."""
        z = (np.asarray(s, dtype=float) - LOW) / (HIGH - LOW) * self.bins
        return np.clip(z.astype(int), 0, self.bins - 1)

    def __call__(self, s):
        i, j = self.cell(s)
        return np.array([i * self.bins + j])


class TileFeatures:
    """tilings сеток n × n, сдвинутых друг относительно друга: в каждой сетке точка лежит в одной плитке."""

    def __init__(self, n=8, tilings=8):
        self.size, self.tilings = n, tilings
        self.per_tiling = (n + 1) ** 2               # сдвинутая сетка вылезает за край на одну плитку
        self.n = tilings * self.per_tiling
        self.offsets = np.array([1, 3]) * np.arange(tilings)[:, None] / tilings

    def tiles(self, s):
        """Для каждой сетки — номера плитки по позиции и по скорости (массив tilings × 2)."""
        z = (np.asarray(s, dtype=float) - LOW) / (HIGH - LOW) * self.size
        return np.clip(np.floor(z + self.offsets).astype(int), 0, self.size)

    def __call__(self, s):
        t = self.tiles(s)
        return np.arange(self.tilings) * self.per_tiling + t[:, 0] * (self.size + 1) + t[:, 1]


# ---------------------------------------------------------------------------------------------- след обновления
def plane(k=121):
    """k × k точек (x, v), покрывающих всё поле наблюдений; возвращает (точки (k·k, 2), форма (k, k))."""
    xs, vs = np.linspace(LOW[0], HIGH[0], k), np.linspace(LOW[1], HIGH[1], k)
    X, V = np.meshgrid(xs, vs)                    # строки — скорость, столбцы — позиция
    return np.column_stack([X.ravel(), V.ravel()]), X.shape


def footprint_linear(features, point, points):
    """Шаг обновления в point меняет веса единиц f(point); оценка в s′ сдвигается пропорционально числу общих единиц."""
    active = set(features(point).tolist())
    return np.array([len(active.intersection(features(p).tolist())) for p in points]) / len(active)


def footprint_net(net, point, points, target=-10.0, lr=0.01):
    """Один шаг градиента ½(target − V(point))² по весам сети; сдвиг оценки во всех точках и в самой point."""
    import torch
    inputs = torch.tensor(normalize(points), dtype=torch.float32)
    x0 = torch.tensor(normalize(point)[None], dtype=torch.float32)
    with torch.no_grad():
        before, before0 = net(inputs).squeeze(1), net(x0).item()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = 0.5 * (target - net(x0).squeeze()) ** 2
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        after, after0 = net(inputs).squeeze(1), net(x0).item()
    return (after - before).numpy(), after0 - before0


# ---------------------------------------------------------------------------------------------- картинки
def _hill(x):
    return 0.45 * np.sin(3 * x) + 0.55


def draw_mountain_car(x=None, ax=None, title=""):
    """Профиль горок, флаг на x = 0.5 и машина (если задана позиция x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = np.linspace(LOW[0], HIGH[0], 300)
    ax.fill_between(xs, 0, _hill(xs), color="#e8e2d0")
    ax.plot(xs, _hill(xs), color="k", lw=1.5)
    ax.plot([FLAG, FLAG], [_hill(FLAG), _hill(FLAG) + 0.22], color="k", lw=1.5)
    ax.add_patch(plt.Polygon([[FLAG, _hill(FLAG) + 0.22], [FLAG + 0.1, _hill(FLAG) + 0.18],
                              [FLAG, _hill(FLAG) + 0.14]], color=C_ACCENT))
    ax.text(FLAG - 0.05, _hill(FLAG) + 0.12, "флаг, x = 0.5", ha="right", fontsize=12, color=C_ACCENT)
    if x is not None:
        y = _hill(x)
        angle = np.degrees(np.arctan(1.35 * np.cos(3 * x)))
        car = Rectangle((x - 0.07, y), 0.14, 0.07, angle=angle, rotation_point=(x, y),
                        facecolor=C_AGENT, edgecolor="k", zorder=3)
        ax.add_patch(car)
        ax.add_patch(FancyArrowPatch((x - 0.38, y + 0.2), (x - 0.12, y + 0.2), arrowstyle="<|-", mutation_scale=16,
                                     color=C_ENV, lw=2))
        ax.add_patch(FancyArrowPatch((x + 0.12, y + 0.2), (x + 0.38, y + 0.2), arrowstyle="-|>", mutation_scale=16,
                                     color=C_ENV, lw=2))
        ax.text(x, y + 0.3, "толкать ←   не толкать   толкать →", ha="center", fontsize=11, color=C_ENV)
    ax.set_xlim(LOW[0], HIGH[0] + 0.05); ax.set_ylim(0, 1.25)
    ax.set_xlabel("позиция x", fontsize=12); ax.set_yticks([])
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def _plane_axes(ax):
    ax.set_xlabel("позиция x", fontsize=12); ax.set_ylabel("скорость v", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.axvline(FLAG, color=C_ACCENT, lw=1.5, ls="--")


def plot_visits(visits, ax=None, title=""):
    """Сколько раз обновлялась каждая клетка сетки; белые клетки не обновлялись ни разу."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.4))
    shown = np.ma.masked_equal(np.asarray(visits, dtype=float).T, 0)   # строки — скорость, столбцы — позиция
    im = ax.imshow(shown, origin="lower", aspect="auto", cmap="Blues", norm=LogNorm(vmin=1, vmax=shown.max()),
                   extent=[LOW[0], HIGH[0], LOW[1], HIGH[1]])
    bins = visits.shape[0]
    for k in range(1, bins):
        ax.axvline(LOW[0] + k * (HIGH[0] - LOW[0]) / bins, color="#dddddd", lw=0.5)
        ax.axhline(LOW[1] + k * (HIGH[1] - LOW[1]) / bins, color="#dddddd", lw=0.5)
    _plane_axes(ax)
    bar = plt.colorbar(im, ax=ax)
    bar.set_label("обновлений клетки", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def draw_tilings(features, point, ax=None, title="", show=3, around=2.5):
    """Линии первых show сеток плиток и плитки этих сеток, в которые попала точка point.

    Показан фрагмент поля: around ширин плитки в каждую сторону от point (на всём поле сдвиги не разглядеть)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.4))
    colors = [C_AGENT, C_REWARD, C_ENV, C_ACCENT, C_GREY]
    width = (HIGH - LOW) / features.size
    tiles = features.tiles(point)
    for t in range(show):
        color = colors[t % len(colors)]
        start = LOW - features.offsets[t] * width
        for k in range(features.size + 2):
            ax.axvline(start[0] + k * width[0], color=color, lw=0.8, alpha=0.6)
            ax.axhline(start[1] + k * width[1], color=color, lw=0.8, alpha=0.6)
        corner = start + tiles[t] * width
        ax.add_patch(Rectangle(corner, width[0], width[1], facecolor=color, alpha=0.25, edgecolor=color, lw=2.5))
        ax.plot([], [], color=color, lw=2.5, label=f"сетка {t + 1}")
    ax.plot(*point, marker="*", ms=16, color="k", zorder=4)
    ax.set_xlim(point[0] - around * width[0], point[0] + around * width[0])
    ax.set_ylim(point[1] - around * width[1], point[1] + around * width[1])
    ax.set_xlabel("позиция x", fontsize=12); ax.set_ylabel("скорость v", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_title(title, fontsize=12)
    return ax


def plot_footprint(ratio, shape, point, ax=None, title=""):
    """Карта следа: во сколько раз сдвинулась оценка в каждой точке по сравнению со сдвигом в самой точке (★)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(np.asarray(ratio).reshape(shape), origin="lower", aspect="auto", cmap="coolwarm",
                   vmin=-1.2, vmax=1.2, extent=[LOW[0], HIGH[0], LOW[1], HIGH[1]])
    ax.plot(*point, marker="*", ms=16, color="k")
    _plane_axes(ax)
    plt.colorbar(im, ax=ax).set_label("сдвиг / сдвиг в ★", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_lengths(curves, ax=None, title="", window=10):
    """Длины эпизодов MountainCar: тонкие линии — запуски, жирная — медиана; 200 шагов — машина не доехала."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [C_REWARD, C_AGENT, C_ENV, C_GREY]
    kernel = np.ones(window) / window
    for (name, runs), color in zip(curves.items(), colors):
        runs = np.asarray(runs, dtype=float)
        smooth = np.array([np.convolve(r, kernel, mode="valid") for r in runs])
        episodes = np.arange(window, runs.shape[1] + 1)
        for run in smooth:
            ax.plot(episodes, run, color=color, lw=1, alpha=0.35)
        ax.plot(episodes, np.median(smooth, axis=0), color=color, lw=3, label=f"{name}: медиана")
    ax.axhline(MAX_STEPS, color="k", ls="--", lw=1, zorder=0)
    ax.text(window, MAX_STEPS + 3, "200 — не доехала до флага", fontsize=11, va="bottom")
    ax.set_ylim(110, 220); ax.grid(alpha=0.3)
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("шагов до флага", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower left", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_w_curves(curves, ax=None, title=""):
    """Оценка V(1) по обновлениям для нескольких вариантов обучения; логарифмическая шкала."""
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
