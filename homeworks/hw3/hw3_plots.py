"""Обвязка и картинки домашки блока 3: коридор с серыми клетками и Acrobot.

Читать этот файл не обязательно: всё, что нужно в домашке, — вызвать функции ниже.
Рендер Acrobot свой, на matplotlib: pygame не нужен.

Коридор с серыми клетками
    CELLS = 6, EXIT = 2               клетки 0…5, выход — клетка 2
    STARTS = (0, 1, 3, 4, 5)          старт — одна из пяти клеток, все равновероятны
    MAX_STEPS = 200                   эпизод обрезается после 200 шагов
    LEFT, RIGHT = 0, 1                номера действий ← и →
    OBS_NAMES                         что видит датчик: 0 — «стена слева», 1 — «серая клетка», 2 — «стена справа»
    observe(cell) -> 0, 1 или 2       наблюдение в клетке
    next_cell(cell, a) -> клетка      куда ведёт действие a; в стену — остаёмся на месте
    GreyCorridor()                    среда; reset(seed=None) -> (наблюдение, {"cell": клетка}),
                                      step(a) -> (наблюдение, −1, дошли до выхода, обрезан, {"cell": клетка})

Рисовальщики (принимают ax=None, рисуют, возвращают ax)
    draw_grey_corridor(ax=None, title="", probs=None)
        probs — вероятности → для трёх наблюдений; подписываются над клетками
    plot_value_curves(ps, curves, ax=None, best=None, title="")
        curves — словарь {подпись: ценность старта при каждом p из ps}; best — точка (p, ценность)
    plot_prob_runs(runs, ax=None, target=None, title="", ylabel="", color=C_AGENT)
        runs — массив (запуски, эпизоды); каждая линия — один запуск; target — пунктир
    draw_acrobot(s, ax=None, title="", action=None)
        s — наблюдение Acrobot (cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2); action — 0, 1, 2
    plot_lengths(curves, ax=None, title="", color=C_AGENT, window=10)
        curves — массив (запуски, эпизоды) длин эпизодов Acrobot; тонкие линии — запуски, жирная — медиана

Анимация
    animate_acrobot(states, actions=None, path="acrobot.gif", every=2, fps=12)
        GIF из списка состояний; возвращает IPython Image — показать через display()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

# --------------------------------------------------------------------------- коридор с серыми клетками
CELLS, EXIT = 6, 2
STARTS = (0, 1, 3, 4, 5)
MAX_STEPS = 200
LEFT, RIGHT = 0, 1
OBS_NAMES = ("стена слева", "серая клетка", "стена справа")
_OBS = {0: 0, 1: 1, 3: 1, 4: 1, 5: 2}


def observe(cell):
    """Что видит датчик в клетке: 0 — стена слева, 1 — серая клетка, 2 — стена справа."""
    return _OBS[cell]


def next_cell(cell, a):
    """Клетка после действия a (0 = ←, 1 = →); шаг в стену оставляет агента на месте."""
    return min(cell + 1, CELLS - 1) if a == RIGHT else max(cell - 1, 0)


class GreyCorridor:
    """Коридор из шести клеток, выход в клетке 2. Награда −1 за каждый шаг, эпизод — до выхода или 200 шагов."""

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.cell = STARTS[int(rng.integers(len(STARTS)))]
        self.steps = 0
        return observe(self.cell), {"cell": self.cell}

    def step(self, a):
        self.cell = next_cell(self.cell, a)
        self.steps += 1
        terminated = self.cell == EXIT
        truncated = not terminated and self.steps >= MAX_STEPS
        obs = 0 if terminated else observe(self.cell)
        return obs, -1.0, terminated, truncated, {"cell": self.cell}


def draw_grey_corridor(ax=None, title="", probs=None):
    """Шесть клеток: серые агент не различает, зелёная — выход; под клетками — что видит датчик."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(-0.3, CELLS + 0.3); ax.set_ylim(-1.0, 1.75); ax.set_aspect("equal"); ax.axis("off")
    for c in range(CELLS):
        color = "#a9dfa9" if c == EXIT else "#c8c8c8" if c in (1, 3, 4) else "white"
        ax.add_patch(Rectangle((c, 0), 1, 1, facecolor=color, edgecolor="#555555", lw=1.2))
        ax.text(c + 0.5, 1.08, str(c), ha="center", va="bottom", fontsize=13)
        label = "выход" if c == EXIT else OBS_NAMES[observe(c)]
        ax.text(c + 0.5, -0.12, label.replace(" ", "\n"), ha="center", va="top", fontsize=11,
                color="#2f6b2f" if c == EXIT else "#333333")
        if probs is not None and c != EXIT:
            ax.text(c + 0.5, 0.5, f"→ {probs[observe(c)]:.2f}", ha="center", va="center", fontsize=11)
    ax.plot([0, 0], [0, 1], color="black", lw=4); ax.plot([CELLS, CELLS], [0, 1], color="black", lw=4)
    ax.set_title(title, fontsize=12)
    return ax


def plot_value_curves(ps, curves, ax=None, best=None, title=""):
    """Ценность старта против вероятности → в серых клетках; по линии на каждую подпись."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.8))
    colors = [C_AGENT, C_REWARD, C_ENV, C_ACCENT]
    for (label, values), color in zip(curves.items(), colors):
        ax.plot(ps, values, color=color, lw=2.5, label=label)
    if best is not None:
        ax.plot(*best, "o", color="black", ms=8, zorder=5)
        ax.annotate(f"$p_G$ = {best[0]:.2f}, ценность {best[1]:.2f}".replace("-", "−"), best,
                    textcoords="offset points", xytext=(10, -22), fontsize=11)
    ax.set_xlabel("$p_G$ — вероятность → в серых клетках", fontsize=12)
    ax.set_ylabel("ценность старта", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="lower center")
    ax.set_title(title, fontsize=12)
    return ax


def plot_prob_runs(runs, ax=None, target=None, title="", ylabel="", color=C_AGENT):
    """Каждая линия — один запуск: как менялась вероятность по эпизодам; target — пунктир с точным ответом."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4))
    for run in np.asarray(runs):
        ax.plot(run, color=color, lw=1.2, alpha=0.5)
    if target is not None:
        ax.axhline(target, color="black", ls="--", lw=1.5, label=f"точный ответ {target:.2f}")
        ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=12)
    return ax


# --------------------------------------------------------------------------- Acrobot
_TORQUE = {0: "момент −1", 1: "момент 0", 2: "момент +1"}


def draw_acrobot(s, ax=None, title="", action=None):
    """Два звена на шарнирах: верхний шарнир закреплён, мотор — в среднем. Цель — поднять конец выше пунктира."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cos1, sin1, cos2, sin2 = np.asarray(s, dtype=float)[:4]
    t1, t2 = np.arctan2(sin1, cos1), np.arctan2(sin2, cos2)
    joint = np.array([np.sin(t1), -np.cos(t1)])
    tip = joint + np.array([np.sin(t1 + t2), -np.cos(t1 + t2)])
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(1.0, color=C_ENV, ls="--", lw=1.5)
    ax.text(2.25, 1.08, "цель", ha="right", va="bottom", fontsize=11, color="#2f6b2f")
    ax.plot([0, joint[0]], [0, joint[1]], color=C_AGENT, lw=6, solid_capstyle="round")
    ax.plot([joint[0], tip[0]], [joint[1], tip[1]], color=C_REWARD, lw=6, solid_capstyle="round")
    ax.plot(0, 0, "ks", ms=8); ax.plot(*joint, "o", color="black", ms=7)
    if action is not None:
        ax.text(joint[0], joint[1] - 0.35, _TORQUE[int(action)], ha="center", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.set_title(title, fontsize=12)
    return ax


def plot_lengths(curves, ax=None, title="", color=C_AGENT, window=10):
    """Сколько шагов понадобилось на подъём в каждом эпизоде: тонкие линии — запуски, жирная — медиана."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    curves = np.asarray(curves, dtype=float)
    kernel = np.ones(window) / window
    smooth = np.array([np.convolve(c, kernel, mode="valid") for c in curves])
    episodes = np.arange(window, curves.shape[1] + 1)
    for run in smooth:
        ax.plot(episodes, run, color=color, lw=1.2, alpha=0.5, zorder=3)
    ax.plot(episodes, np.median(smooth, axis=0), color=color, lw=3, label="медиана по запускам", zorder=4)
    ax.axhline(500, color=C_GREY, ls="--", lw=1, zorder=1)
    ax.text(episodes[-1], 485, "лимит 500", fontsize=11, color=C_GREY, va="top", ha="right")
    ax.set_ylim(0, 530); ax.grid(alpha=0.3)
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("шагов до подъёма", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower left", fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def animate_acrobot(states, actions=None, path="acrobot.gif", every=2, fps=12):
    """GIF эпизода по списку состояний (кадр на каждый every-й шаг); возвращает IPython Image."""
    import io
    from PIL import Image as PILImage
    from IPython.display import Image

    frames = []
    steps = list(range(0, len(states), every))
    if steps[-1] != len(states) - 1:
        steps.append(len(states) - 1)
    for k in steps:
        fig, ax = plt.subplots(figsize=(4, 4))
        action = actions[k] if actions is not None and k < len(actions) else None
        draw_acrobot(states[k], ax=ax, title=f"шаг {k}", action=action)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).convert("P", palette=PILImage.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    with open(path, "rb") as f:
        return Image(data=f.read(), format="gif")
