"""Картинки семинара 5: тележка с шестом (CartPole), кривые обучения, карта стратегии.

Читать этот файл не обязательно: всё, что нужно на семинаре, — вызвать функции ниже.
Свой рендер на matplotlib: pygame не нужен.

Рисовальщики (принимают ax=None, рисуют, возвращают ax)
    draw_cartpole(s, ax=None, title="", action=None)
        s — состояние (x, ẋ, φ, φ̇); action — 0 (действие ←) или 1 (действие →), рисуется стрелкой
    plot_lengths(curves, ax=None, title="", color=C_AGENT)
        curves — массив (запуски, эпизоды) длин эпизодов; тонкие линии — запуски, жирная — медиана
    plot_policy_map(w, ax=None, title="")
        p(→) при весах w на плоскости (угол φ, угловая скорость φ̇), тележка в центре и стоит (x = ẋ = 0)
    plot_vs(x, curves, ax=None, xlabel="", ylabel="", title="")
        curves — словарь {подпись: значения при каждом x}; метрика против параметра

Анимация
    animate_cartpole(states, path="cartpole.gif", every=4, fps=12)
        GIF из списка состояний (кадр на каждый every-й шаг); возвращает IPython Image — показать через display()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"

X_LIMIT = 2.4                     # тележка дальше ±2.4 — конец эпизода
ANGLE_LIMIT = 12 * np.pi / 180    # шест дальше ±12° (0.21 рад) — конец эпизода
POLE_LENGTH = 1.0


def draw_cartpole(s, ax=None, title="", action=None):
    """Тележка на рельсе и шест под углом φ к вертикали (плюс — наклон вправо)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.0))
    x, _, phi, _ = np.asarray(s, dtype=float)
    ax.set_xlim(-X_LIMIT - 0.5, X_LIMIT + 0.5); ax.set_ylim(-0.45, 1.35)
    ax.set_aspect("equal"); ax.set_xticks([-2.4, 0, 2.4]); ax.set_yticks([])
    ax.tick_params(labelsize=11)
    ax.axhline(-0.125, color=C_GREY, lw=1.5)
    for edge in (-X_LIMIT, X_LIMIT):
        ax.axvline(edge, color=C_ACCENT, ls="--", lw=1.2)
    ax.add_patch(Rectangle((x - 0.25, -0.125), 0.5, 0.25, facecolor=C_AGENT, edgecolor="black", lw=1))
    top = np.array([x, 0.125])
    tip = top + POLE_LENGTH * np.array([np.sin(phi), np.cos(phi)])
    ax.plot([top[0], tip[0]], [top[1], tip[1]], color=C_REWARD, lw=6, solid_capstyle="round")
    ax.plot(*top, "o", color="black", ms=5)
    if action is not None:
        sign = 1 if action == 1 else -1
        start = x - sign * 0.95
        ax.add_patch(FancyArrowPatch((start, 0.0), (x - sign * 0.3, 0.0), arrowstyle="-|>",
                                     mutation_scale=18, color=C_ENV, lw=2.5))
    ax.set_title(title, fontsize=12)
    return ax


def plot_lengths(curves, ax=None, title="", color=C_AGENT, window=10):
    """Длина эпизода по ходу обучения: каждый запуск — тонкая линия, медиана по запускам — жирная.

    Кривые сглажены скользящим средним по window эпизодам, иначе они дрожат от эпизода к эпизоду."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
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


def plot_policy_map(w, ax=None, title=""):
    """p(→) на плоскости (угол φ, угловая скорость φ̇) при x = ẋ = 0: красное — действие →, синее — действие ←.

    Сплошная линия — p = 0.5; пунктир — граница эвристики лекции 1 «толкать туда, куда падает шест» (φ̇ = 0)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    w = np.asarray(w, dtype=float)
    phis = np.linspace(-ANGLE_LIMIT, ANGLE_LIMIT, 121)
    phi_dots = np.linspace(-2.0, 2.0, 121)
    P, D = np.meshgrid(phis, phi_dots)
    h = np.clip(w[2] * P + w[3] * D, -50, 50)
    p = 1 / (1 + np.exp(-h))
    im = ax.imshow(p, origin="lower", aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
                   extent=(phis[0], phis[-1], phi_dots[0], phi_dots[-1]))
    ax.contour(P, D, p, levels=[0.5], colors="black", linewidths=1.5)
    ax.axhline(0, color="black", ls=":", lw=1.5)
    ax.text(phis[0] + 0.005, -0.08, "правило лекции 1: граница φ̇ = 0", fontsize=11, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("p(→)", fontsize=12)
    ax.set_xlabel("угол шеста φ, рад (плюс — наклон вправо)", fontsize=12)
    ax.set_ylabel("угловая скорость φ̇, рад/с", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def plot_vs(x, curves, ax=None, xlabel="", ylabel="", title=""):
    """Метрика против параметра: по линии с точками на каждую подпись; ось x логарифмическая."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    colors = [C_REWARD, C_ENV, C_AGENT, C_ACCENT]
    for (label, values), color in zip(curves.items(), colors):
        ax.plot(x, values, "-o", color=color, lw=2.5, ms=7, label=label)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0, 520); ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=12)
    return ax


def animate_cartpole(states, path="cartpole.gif", every=4, fps=12):
    """GIF эпизода по списку состояний (кадр на каждый every-й шаг); возвращает IPython Image."""
    import io
    from PIL import Image as PILImage
    from IPython.display import Image

    frames = []
    steps = list(range(0, len(states), every))
    if steps[-1] != len(states) - 1:
        steps.append(len(states) - 1)
    for k in steps:
        fig, ax = plt.subplots(figsize=(5.4, 2.7))
        draw_cartpole(states[k], ax=ax, title=f"шаг {k}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).convert("P", palette=PILImage.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    with open(path, "rb") as f:
        return Image(data=f.read(), format="gif")
