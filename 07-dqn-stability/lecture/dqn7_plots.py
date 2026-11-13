"""Картинки лекции 7: схема обучения Q-сети, смертельная триада, оценка против настоящей суммы наград, DQN на Atari.

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.
Рисовальщики принимают ax=None, рисуют и возвращают ax.

    draw_dqn_scheme(target=False, ax=None)
        схема одного шага обучения: среда → буфер → пачка → оценка и цель → шаг градиента.
        target=False — цель считает сама обучаемая сеть (красная петля); target=True — её медленная копия.
    draw_triad(ax=None)
        три условия — три круга; алгоритмы курса расставлены по областям, в центре — все три условия сразу.
    draw_two_states_scheme(ax=None)             пример «w и 2w»: два состояния, один вес, какие переходы есть в данных
    plot_chase_frame(axes, history, phase, k)   один кадр анимации «оценка тянется к цели» (assets/make_animations.py)
    plot_toy_curves(curves, ax=None, title="")  пример «w и 2w»: оценка по шагам, curves — словарь {подпись: массив чисел}
    plot_estimate_vs_real(q_start, lengths, ax=None, title="", window=10)
        оценка старта max Q и сумма наград, которую эпизод принёс на самом деле; q_start и lengths —
        массивы (запуски, эпизоды); рисуется среднее по запускам, завышение закрашено.
    plot_max_bias(samples, ax=None, title="")
        гистограммы оценок: samples — словарь {подпись: массив чисел}; у каждой подписана средняя.
    draw_atari_scheme(ax=None)                  DQN на Atari: четыре кадра → свёрточная сеть → Q для каждой кнопки
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"


def _box(ax, x, y, w, h, text, color, face="#ffffff", size=12, weight="normal", lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def _arrow(ax, a, b, color="k", lw=2.2, rad=0.0, ls="-", scale=20):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def draw_dqn_scheme(target=False, ax=None):
    """Один шаг обучения Q-сети. Различие двух вариантов — только в том, какие веса считают цель y."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5.4))
    # среда и буфер
    _box(ax, 0.0, 3.3, 2.3, 1.1, "среда\nCartPole", C_ENV, size=13, weight="bold")
    _box(ax, 4.0, 3.3, 2.7, 1.1, "буфер опыта\n50 000 переходов", C_ENV, size=12)
    _arrow(ax, (2.4, 3.85), (3.9, 3.85), C_ENV)
    ax.text(3.15, 4.08, "переход", ha="center", fontsize=11, color=C_ENV)
    ax.text(3.15, 3.42, "s, a, r, s′,\nконец", ha="center", va="top", fontsize=10, color=C_ENV)
    # сети
    cut = C_REWARD if target else C_ACCENT
    _box(ax, 9.6, 4.3, 2.6, 1.0, "Q-сеть, веса $\\theta$", C_AGENT, face="#dfe7f3", size=13, weight="bold")
    if target:
        _box(ax, 9.6, 1.5, 2.6, 1.0, "копия, веса $\\bar\\theta$", cut, face="#fbe9dc", size=13, weight="bold")
    else:
        _box(ax, 9.6, 1.5, 2.6, 1.0, "та же Q-сеть,\nвеса $\\theta$", cut, face="#f7dfe0", size=12, weight="bold")
    _arrow(ax, (6.85, 4.1), (9.5, 4.75), "k")
    _arrow(ax, (6.85, 3.6), (9.5, 2.05), "k")
    ax.text(7.3, 3.85, "пачка: 64 перехода", ha="left", va="center", fontsize=11)
    ax.text(7.9, 4.72, "s, a", fontsize=12, style="italic"); ax.text(7.7, 2.3, "r, s′", fontsize=12, style="italic")
    # оценка и цель
    _box(ax, 13.2, 4.3, 2.4, 1.0, "оценка\n$Q_\\theta(s, a)$", C_AGENT, size=12)
    weights = "\\bar\\theta" if target else "\\theta"
    _box(ax, 13.2, 1.5, 3.5, 1.0, f"цель  $y = r + \\gamma \\max Q_{{{weights}}}(s')$", cut, size=12)
    _arrow(ax, (12.3, 4.8), (13.1, 4.8), C_AGENT); _arrow(ax, (12.3, 2.0), (13.1, 2.0), cut)
    # ошибка и шаг
    _box(ax, 17.4, 2.9, 2.4, 1.0, "ошибка\n$(y - Q_\\theta(s, a))^2$", "k", size=12)
    _arrow(ax, (15.7, 4.6), (17.5, 4.0), "k"); _arrow(ax, (16.8, 2.3), (17.7, 2.85), "k")
    _arrow(ax, (18.6, 4.0), (11.8, 5.4), C_AGENT, rad=0.28, lw=2.6)
    ax.text(16.0, 6.3, "шаг градиента меняет веса $\\theta$", ha="center", fontsize=12, color=C_AGENT, weight="bold")
    # действие
    _arrow(ax, (9.6, 5.1), (1.2, 4.5), C_AGENT, rad=0.2, lw=1.8)
    ax.text(5.0, 5.85, "действие: с наибольшим Q, иногда случайное", ha="center", fontsize=11, color=C_AGENT)
    if target:
        _arrow(ax, (10.9, 4.25), (10.9, 2.6), cut, lw=2.2, ls=(0, (4, 3)))
        ax.text(11.05, 3.42, "каждый шаг копия\nсдвигается к $\\theta$ на 1 %", ha="left", va="center", fontsize=11, color=cut)
        ax.text(13.0, 0.75, "шаг градиента копию не трогает — цель почти стоит на месте", ha="center",
                fontsize=12.5, color=cut, weight="bold")
    else:
        _arrow(ax, (10.9, 4.25), (10.9, 2.6), cut, lw=3.0)
        ax.text(11.05, 3.42, "те же веса:\nшаг сдвигает и цель", ha="left", va="center", fontsize=11.5, color=cut,
                weight="bold")
        ax.text(13.0, 0.75, "цель считает сеть, которую мы учим: каждый шаг её двигает", ha="center",
                fontsize=12.5, color=cut, weight="bold")
    ax.set_xlim(-0.3, 20.1); ax.set_ylim(0.3, 6.8); ax.axis("off")
    return ax


def draw_triad(ax=None):
    """Смертельная триада кругами: каждое условие — круг, алгоритмы стоят в области своих условий."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12.5, 7.6))
    r = 2.35
    centres = {"approx": (0.0, 1.45), "boot": (-1.4, -0.95), "off": (1.4, -0.95)}
    colours = {"approx": C_AGENT, "boot": C_REWARD, "off": C_ENV}
    for key, (x, y) in centres.items():
        ax.add_patch(Circle((x, y), r, facecolor=colours[key], alpha=0.16, edgecolor="none"))
        ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=colours[key], lw=2.5))
    ax.text(0.0, 4.75, "аппроксимация", ha="center", fontsize=15, weight="bold", color=C_AGENT)
    ax.text(0.0, 4.42, "одни веса на много состояний", ha="center", va="top", fontsize=12, color=C_AGENT)
    ax.text(-4.1, -1.7, "бутстрэппинг", ha="right", fontsize=15, weight="bold", color=C_REWARD)
    ax.text(-4.1, -2.03, "в цели — своя оценка $Q(s')$", ha="right", va="top", fontsize=12, color=C_REWARD)
    ax.text(4.1, -1.7, "off-policy", ha="left", fontsize=15, weight="bold", color=C_ENV)
    ax.text(4.1, -2.03, "данные собрала другая стратегия", ha="left", va="top", fontsize=12, color=C_ENV)
    label = dict(ha="center", va="center", fontsize=12)
    ax.text(0.0, 2.75, "REINFORCE\n(лекция 5)", **label)
    ax.text(-2.45, -1.6, "SARSA\nпо таблице", **label)
    ax.text(0.0, -1.95, "Q-learning\nпо таблице", **label)
    ax.text(-1.5, 0.75, "Q-сеть\nпо шагу\nподряд", **label)
    ax.text(0.0, 0.05, "Q-сеть\n+ буфер", ha="center", va="center", fontsize=13, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=C_ACCENT, edgecolor="none"))
    ax.annotate("все три условия сразу:\nоценки могут разлететься", xy=(0.55, 0.05), xytext=(4.3, 1.9), fontsize=12.5,
                color=C_ACCENT, weight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=2, connectionstyle="arc3,rad=0.2"))
    ax.set_xlim(-8.2, 8.6); ax.set_ylim(-3.6, 5.3); ax.set_aspect("equal"); ax.axis("off")
    return ax


def draw_two_states_scheme(ax=None):
    """Пример «w и 2w»: два состояния, один вес w задаёт обе оценки; в данных есть только переход 1 → 2."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.6))
    for x, name in ((1.4, "1"), (4.4, "2")):
        ax.add_patch(Circle((x, 1.2), 0.55, facecolor="#ffffff", edgecolor="k", lw=2))
        ax.text(x, 1.2, name, ha="center", va="center", fontsize=20, weight="bold")
    _arrow(ax, (2.0, 1.2), (3.8, 1.2), C_ENV, lw=2.6)
    ax.text(2.9, 1.5, "награда 0", ha="center", fontsize=13, color=C_ENV)
    ax.text(2.9, 0.88, "есть в данных", ha="center", va="top", fontsize=12, color=C_ENV)
    _arrow(ax, (5.0, 1.2), (6.8, 1.2), C_GREY, lw=2.6, ls=(0, (4, 3)))
    ax.text(5.9, 1.5, "награда 0", ha="center", fontsize=13, color=C_GREY)
    ax.text(5.9, 0.88, "в данных нет", ha="center", va="top", fontsize=12, color=C_GREY)
    ax.text(7.0, 1.2, "конец", va="center", fontsize=14, color=C_GREY)
    ax.text(1.4, 2.15, "оценка $\\hat v(1) = w$", ha="center", fontsize=15, color=C_AGENT)
    ax.text(4.4, 2.15, "оценка $\\hat v(2) = 2w$", ha="center", fontsize=15, color=C_AGENT)
    ax.text(4.2, 2.75, "один вес $w$ задаёт обе оценки; настоящая ценность обоих состояний — 0", ha="center",
            fontsize=13, color=C_AGENT)
    ax.set_xlim(-0.6, 9.0); ax.set_ylim(0.3, 3.1); ax.axis("off")
    return ax


def plot_chase_frame(axes, history, phase, k):
    """Кадр анимации: слева одна ручка w на оба состояния, справа таблица. history[name] — список пар (оценка, цель)
    по шагам; phase 0 — оценка только что подтянулась к цели, phase 1 — цель пересчитана."""
    titles = {"общий вес": "один вес на оба состояния: цель убегает",
              "таблица": "таблица, у состояния 2 своё число: цель стоит"}
    for ax, (name, pairs) in zip(axes, history.items()):
        ax.clear()
        value, goal = pairs[k][0], (pairs[k][1] if phase else pairs[max(k - 1, 0)][1])
        ax.bar([0], [value], 0.5, color=C_AGENT); ax.bar([1], [goal], 0.5, color=C_REWARD)
        ax.text(0, value + 0.08, f"{value:.2f}", ha="center", fontsize=14, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.text(1, goal + 0.08, f"{goal:.2f}", ha="center", fontsize=14, color=C_REWARD)
        ax.plot([-0.25, 1.25], [goal, goal], color=C_REWARD, lw=1.2, ls=":")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["оценка $\\hat v(1)$", "цель $\\gamma\\,\\hat v(2)$"], fontsize=13)
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 5.2); ax.tick_params(axis="y", labelsize=11); ax.grid(axis="y", alpha=0.3)
        ax.set_title(titles[name], fontsize=13, color=C_ACCENT if name == "общий вес" else C_ENV, weight="bold")
    return axes


def plot_toy_curves(curves, ax=None, title=""):
    """Пример «w и 2w»: оценка v̂(1) по шагам для нескольких вариантов; curves — словарь {подпись: массив чисел}.
    Шкала логарифмическая; всё, что меньше 0.001, считаем дошедшим до нуля. Подписи — справа от картинки."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.2))
    colours = [C_ACCENT, C_AGENT, C_ENV, C_REWARD]
    for (name, values), colour in zip(curves.items(), colours):
        last = values[-1]
        shown = "почти 0" if abs(last) < 1e-3 else (f"{last:.0f}" if abs(last) >= 100 else f"{last:.2g}")
        ax.plot(np.arange(1, len(values) + 1), np.abs(values), color=colour, lw=3, label=f"{name}: {shown}")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e4); ax.set_xlim(0, len(values))
    ax.axhline(1, color=C_GREY, lw=1, ls=":")
    ax.text(len(values) * 0.99, 1.15e-3, "ниже — оценка у нуля", ha="right", va="bottom", fontsize=11, color=C_GREY)
    ax.set_xlabel("шаг обучения", fontsize=12); ax.set_ylabel("оценка $\\hat v(1)$", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3, which="major")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=False,
              title="оценка после 100 шагов", title_fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.figure.tight_layout()
    return ax


def plot_estimate_vs_real(q_start, lengths, ax=None, title="", window=10, gamma=0.99):
    """Оценка старта max Q и сумма наград, которую эпизод принёс на самом деле (среднее по запускам)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    kernel = np.ones(window) / window
    smooth = lambda rows: np.convolve(np.mean(rows, axis=0), kernel, mode="valid")
    real = (1 - gamma ** np.asarray(lengths, dtype=float)) / (1 - gamma)
    promised, got = smooth(q_start), smooth(real)
    episodes = np.arange(window - 1, window - 1 + len(got))
    ax.fill_between(episodes, got, promised, where=promised > got, color=C_ACCENT, alpha=0.25, label="завышение")
    ax.plot(episodes, promised, color=C_ACCENT, lw=2.5, label="оценка старта max Q")
    ax.plot(episodes, got, color=C_ENV, lw=2.5, label="сумма наград эпизода")
    ax.axhline(1 / (1 - gamma), color=C_GREY, lw=1.2, ls="--")
    ax.text(episodes[0], 1 / (1 - gamma) + 2, "больше 100 не бывает", fontsize=11, color=C_GREY, va="bottom")
    ax.set_xlabel("эпизод", fontsize=12); ax.set_ylabel("сумма наград с дисконтом", fontsize=12)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=11)
    ax.set_title(title, fontsize=13)
    return ax


def plot_max_bias(samples, ax=None, title=""):
    """Гистограммы оценок с подписанными средними; настоящая ценность — 0 (пунктир)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.4))
    colours = [C_GREY, C_ACCENT, C_ENV]
    bins = np.linspace(-3.5, 3.5, 57)
    for (name, values), colour in zip(samples.items(), colours):
        ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.25, color=colour, density=True)
        ax.hist(values, bins=bins, histtype="step", lw=2.2, color=colour, density=True,
                label=f"{name}: в среднем {np.mean(values):+.2f}")
        ax.axvline(np.mean(values), color=colour, lw=2, ls="-" if abs(np.mean(values)) > 0.05 else ":")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.75)                               # место под подписи
    ax.set_xlabel("оценка (настоящая ценность — 0)", fontsize=12); ax.set_yticks([])
    ax.tick_params(labelsize=11); ax.legend(loc="upper left", fontsize=11.5, framealpha=0.95)
    ax.set_title(title, fontsize=13)
    return ax


def draw_atari_scheme(ax=None):
    """DQN на Atari: четыре последних кадра → свёрточная сеть → Q для каждой кнопки джойстика."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4.2))
    for k in range(4):                                                   # стопка кадров
        ax.add_patch(Rectangle((0.2 + 0.22 * k, 1.0 + 0.2 * k), 1.9, 1.9, facecolor="#20242c", edgecolor="white", lw=1.5))
    ax.add_patch(Rectangle((1.1, 1.75), 0.9, 0.12, facecolor=C_REWARD))  # ракетка
    ax.add_patch(Circle((1.75, 2.75), 0.09, facecolor="white"))          # мяч
    for j in range(6):
        ax.add_patch(Rectangle((0.95 + 0.3 * j, 3.15), 0.25, 0.12, facecolor=[C_ACCENT, C_ENV, C_AGENT][j % 3]))
    ax.text(1.5, 0.55, "состояние: 4 последних кадра\n84 × 84 пикселя", ha="center", va="top", fontsize=12)
    _arrow(ax, (3.1, 2.3), (4.2, 2.3), "k")
    for k, (w, h) in enumerate(((0.55, 2.6), (0.45, 2.0), (0.38, 1.4))):  # свёртки
        ax.add_patch(Rectangle((4.4 + 0.85 * k, 2.3 - h / 2), w, h, facecolor="#dfe7f3", edgecolor=C_AGENT, lw=2))
    for k in range(2):                                                    # полносвязные слои
        ax.add_patch(Rectangle((7.1 + 0.6 * k, 2.3 - 1.1), 0.22, 2.2, facecolor="#dfe7f3", edgecolor=C_AGENT, lw=2))
    ax.text(6.0, 0.55, "Q-сеть: три свёрточных слоя\nи два обычных", ha="center", va="top", fontsize=12, color=C_AGENT)
    _arrow(ax, (8.2, 2.3), (9.3, 2.3), "k")
    values = [0.5, 0.8, 0.35, 1.5, 0.6, 0.45]
    for k, v in enumerate(values):
        ax.add_patch(Rectangle((9.6 + 0.42 * k, 1.3), 0.32, 1.25 * v, facecolor=C_ACCENT if v == max(values) else C_GREY))
    ax.plot([9.5, 12.2], [1.3, 1.3], color="k", lw=1)
    ax.text(10.85, 0.55, "Q для каждой кнопки джойстика\n(от 4 до 18 чисел)", ha="center", va="top", fontsize=12)
    _arrow(ax, (12.4, 2.3), (13.5, 2.3), C_ACCENT)
    ax.text(13.65, 2.3, "нажать кнопку\nс наибольшим Q", va="center", fontsize=12.5, color=C_ACCENT, weight="bold")
    ax.set_xlim(-0.1, 16.2); ax.set_ylim(-0.5, 4.3); ax.axis("off")
    return ax
