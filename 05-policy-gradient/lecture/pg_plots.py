"""Среда и картинки лекции 5: коридор с перепутанной клеткой, softmax, кривые обучения CartPole.

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.

Среда
    Corridor()                    коридор из трёх клеток и выхода; интерфейс как у GridWorld лекции 2:
                                  reset(seed=None) -> (наблюдение, {}), step(a) -> (наблюдение, награда, конец, False, info).
                                  Наблюдение всегда 0: датчик не различает клеток. Настоящая клетка — env.cell и info["cell"].
    LEFT, RIGHT = 0, 1            номера действий
    corridor_value(p)             ценность старта при вероятности → p — точное решение уравнения
                                  Беллмана для стратегии (три уравнения, np.linalg.solve)
    corridor_slope(theta)         наклон ценности старта по θ, где p = e^θ / (1 + e^θ): насколько меняется
                                  ценность при малом сдвиге θ (центральная разность по corridor_value)
    softmax(theta)                предпочтения -> вероятности

Рисовальщики (принимают ax=None, рисуют, возвращают ax)
    draw_corridor(ax=None, title="", sensor=True)
    plot_value_curve(points=None, ax=None, title="", trail=None, marker=None)
        points — словарь {подпись: p}; trail — массив p по эпизодам; marker — текущее p (для анимации)
    plot_probs(rows, ax=None, labels=None, title="", ylim=(0, 1), legend_loc="upper left")
        rows — словарь {подпись: вероятности}; рисует столбики рядом, как «было/стало» в лекции 1
    plot_uphill(ns, shares, ax=None, title="")
        shares — словарь {подпись: доли «оценка указывает в гору»} для числа усредняемых попыток ns
    plot_seeds(curves, ax=None, title="", ylabel="", color=C_AGENT, xlabel="попытка")
        curves — массив (запуски, шаги); каждая линия — один запуск
    plot_lengths(curves, ax=None, title="", color=C_ENV, window=10)
        curves — массив (запуски, эпизоды) длин эпизодов CartPole; тонкие линии — запуски, жирная — медиана
    plot_weights(weights, ax=None, title="")
        weights — вес каждого шага эпизода; столбики: зелёные — положительный вес, красные — отрицательный
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

LEFT, RIGHT = 0, 1
N_CELLS = 3                       # клетки 0, 1, 2; клетка 3 — выход
SWAPPED = 1                       # в этой клетке → ведёт в клетку 0, а ← — в клетку 2


class Corridor:
    """Коридор: старт в клетке 0, выход справа от клетки 2, −1 за каждый шаг, в клетке 1 направления перепутаны."""

    n_actions = 2

    def __init__(self):
        self.cell = 0

    @staticmethod
    def move(cell, action):
        """Куда ведёт действие из клетки (без случайности). В левую стену — остаёмся на месте."""
        step = +1 if action == RIGHT else -1
        if cell == SWAPPED:
            step = -step
        return max(0, cell + step)

    def reset(self, seed=None):
        self.cell = 0
        return 0, {"cell": self.cell}

    def step(self, action):
        self.cell = self.move(self.cell, action)
        terminated = self.cell == N_CELLS
        return 0, -1.0, terminated, False, {"cell": self.cell}


def corridor_value(p):
    """Ценность старта, если в каждой клетке → выбирается с вероятностью p.

    Уравнение Беллмана для стратегии из лекции 2: V(s) = −1 + p·V(куда ведёт →) + (1 − p)·V(куда ведёт ←),
    у выхода V = 0. Три линейных уравнения на V(0), V(1), V(2). При p = 0 или 1 выхода нет: возвращаем −inf.
    """
    if p <= 0.0 or p >= 1.0:
        return -np.inf
    A, b = np.eye(N_CELLS), -np.ones(N_CELLS)
    for c in range(N_CELLS):
        for action, prob in ((RIGHT, p), (LEFT, 1 - p)):
            nxt = Corridor.move(c, action)
            if nxt < N_CELLS:
                A[c, nxt] -= prob
    return float(np.linalg.solve(A, b)[0])


def corridor_slope(theta, h=1e-5):
    """Наклон ценности старта по предпочтению → θ (p = e^θ / (1 + e^θ)) — эталон для сравнения с оценкой REINFORCE."""
    def value(t):
        return corridor_value(1.0 / (1.0 + np.exp(-t)))
    return (value(theta + h) - value(theta - h)) / (2 * h)


def softmax(theta):
    """Вероятности e^θ / Σ e^θ (с вычитанием максимума, чтобы экспонента не переполнялась)."""
    z = np.exp(np.asarray(theta, dtype=float) - np.max(theta))
    return z / z.sum()


# ---------------------------------------------------------------------------------------------- картинки
def draw_corridor(ax=None, title="", sensor=True):
    """Схема коридора: три клетки и выход; в каждой клетке стрелками — куда на самом деле ведут оба действия."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.set_xlim(-0.05, 4.05); ax.set_ylim(-0.85, 1.45); ax.set_aspect("equal"); ax.axis("off")
    for c in range(N_CELLS + 1):
        exit_cell = c == N_CELLS
        face = "#a9dfa9" if exit_cell else ("#fde8d0" if c == SWAPPED else "#ffffff")
        ax.add_patch(Rectangle((c, 0), 1, 1, facecolor=face, edgecolor="k", lw=1.2))
        ax.text(c + 0.5, 0.86, "выход" if exit_cell else f"клетка {c}", ha="center", va="center", fontsize=12)
    ax.text(0.1, 0.12, "S", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(3.5, 0.45, "G", ha="center", va="center", fontsize=22, weight="bold", color="#2f6b2f")
    for c in range(N_CELLS):
        for action, color, y in ((RIGHT, C_AGENT, 0.60), (LEFT, C_REWARD, 0.32)):
            step = +1 if Corridor.move(c, action) > c else -1   # в стену (клетка не меняется) — стрелка влево
            x0, x1 = c + 0.5 - 0.22 * step, c + 0.5 + 0.22 * step
            ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16, lw=2.4, color=color))
    ax.text(SWAPPED + 0.5, -0.08, "направления\nперепутаны", ha="center", va="top", fontsize=11, color=C_ACCENT)
    ax.text(0.5, -0.08, "← — в стену", ha="center", va="top", fontsize=10, color=C_GREY)
    ax.plot([], [], color=C_AGENT, lw=2.4, label="куда ведёт →")
    ax.plot([], [], color=C_REWARD, lw=2.4, label="куда ведёт ←")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2, fontsize=11, frameon=False)
    if sensor:
        ax.text(2.0, -0.62, "награда −1 за каждый шаг\nдатчик агента во всех клетках показывает одно и то же",
                ha="center", va="center", fontsize=11, color=C_GREY)
    ax.set_title(title, fontsize=12)
    return ax


def plot_value_curve(points=None, ax=None, title="", trail=None, marker=None):
    """Кривая «ценность старта против p» с отмеченными стратегиями, след обучения и текущая точка."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.8))
    ps = np.linspace(0.02, 0.98, 481)
    ax.plot(ps, [corridor_value(p) for p in ps], color=C_GREY, lw=2)
    best = 2 - np.sqrt(2)
    ax.axvline(best, color=C_ACCENT, lw=1, ls=":")
    ax.text(best + 0.01, -96, "лучшая p ≈ 0.59", color=C_ACCENT, fontsize=10, ha="left", va="bottom")
    if trail is not None:
        trail = np.asarray(trail)
        ax.plot(trail, [corridor_value(p) for p in trail], ".", color=C_AGENT, ms=4, alpha=0.35)
    if marker is not None:
        ax.plot([marker], [corridor_value(marker)], "o", color=C_AGENT, ms=12, mec="k")
    for label, p in (points or {}).items():
        v = corridor_value(p)
        if not np.isfinite(v):
            continue
        ax.plot([p], [v], "o", color=C_ACCENT, ms=8)
        if v < -60:                                    # внизу кривой — подпись справа от точки
            offset, ha, va = (10, 0), "left", "center"
        elif p > 0.8:                                  # правый склон — подпись над точкой
            offset, ha, va = (0, 10), "center", "bottom"
        else:                                          # у вершины — под точкой, по разные стороны от лучшей
            offset, ha, va = ((-4, -10), "right", "top") if p < best - 1e-6 else ((4, -10), "left", "top")
        ax.annotate(f"{label}\n{v:.1f}".replace("\n-", "\n−"), (p, v), textcoords="offset points",
                    xytext=offset, ha=ha, va=va, fontsize=11)
    ax.set_ylim(-100, 0); ax.set_xlim(0, 1)
    ax.set_xlabel("p — вероятность действия →", fontsize=12)
    ax.set_ylabel("ценность старта", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=12)
    return ax


def plot_probs(rows, ax=None, labels=None, title="", ylim=(0, 1), legend_loc="upper left"):
    """Столбики вероятностей нескольких стратегий рядом (как «клетка S: было / стало» в лекции 1)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.2))
    rows = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    n = len(next(iter(rows.values())))
    labels = labels or [str(i + 1) for i in range(n)]
    width = 0.8 / len(rows)
    colors = [C_GREY, C_AGENT, C_REWARD, C_ENV]
    for k, (name, probs) in enumerate(rows.items()):
        x = np.arange(n) + (k - (len(rows) - 1) / 2) * width
        bars = ax.bar(x, probs, width=width * 0.95, color=colors[k % len(colors)], label=name)
        for xi, v in zip(x, probs):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(*ylim); ax.set_ylabel("вероятность", fontsize=11)
    ax.legend(fontsize=10, loc=legend_loc, frameon=False)
    ax.set_title(title, fontsize=12)
    return ax


def plot_uphill(ns, shares, ax=None, title=""):
    """Доля оценок, указывающих в гору, против числа усреднённых попыток (ось x логарифмическая)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = [C_AGENT, C_REWARD, C_ENV, C_ACCENT]
    for k, (name, ys) in enumerate(shares.items()):
        ax.plot(ns, ys, "o-", color=colors[k % len(colors)], lw=2, label=name)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.axhline(0.5, color=C_GREY, lw=1, ls=":")
    ax.set_ylim(0.2, 1.03)
    ax.set_xlabel("сколько попыток усреднили", fontsize=12)
    ax.set_ylabel("доля оценок «в гору»", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
    ax.set_title(title, fontsize=12)
    return ax


def plot_seeds(curves, ax=None, title="", ylabel="", color=C_AGENT, xlabel="попытка"):
    """Каждая линия — один запуск; полупрозрачные, чтобы было видно, куда приходит большинство."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    for run in np.asarray(curves):
        ax.plot(run, color=color, lw=1.2, alpha=0.45)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=12)
    return ax


def plot_lengths(curves, ax=None, title="", color=C_ENV, window=10):
    """Сколько шагов простоял шест в каждом эпизоде CartPole: тонкие линии — запуски, жирная — медиана.

    Кривые сглажены скользящим средним по window эпизодам, иначе они дрожат от эпизода к эпизоду."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    curves = np.asarray(curves, dtype=float)
    kernel = np.ones(window) / window
    smooth = np.array([np.convolve(c, kernel, mode="valid") for c in curves])
    episodes = np.arange(window, curves.shape[1] + 1)
    for run in smooth:
        ax.plot(episodes, run, color=color, lw=1, alpha=0.35)
    ax.plot(episodes, np.median(smooth, axis=0), color=color, lw=3, label="медиана по запускам")
    ax.axhline(500, color=C_GREY, ls="--", lw=1)
    ax.text(episodes[0], 485, "максимум 500", fontsize=11, color=C_GREY, va="top")
    ax.set_ylim(0, 520); ax.grid(alpha=0.3)
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("сколько шагов простоял шест", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_weights(weights, ax=None, title=""):
    """Вес каждого шага эпизода столбиком: зелёный — шаг поднимает вероятность сыгранного действия, красный — опускает."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    weights = np.asarray(weights, dtype=float)
    colors = [C_ENV if w > 0 else C_ACCENT for w in weights]
    ax.bar(np.arange(len(weights)), weights, color=colors, width=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("шаг эпизода t", fontsize=12); ax.set_ylabel("вес толчка", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=12)
    return ax
