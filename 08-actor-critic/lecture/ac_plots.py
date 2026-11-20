"""Картинки лекции 8 «Актор и критик».

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.
Рисовальщики принимают ax=None, рисуют и возвращают ax.

    draw_ac_scheme(stage, ax=None)
        схема одного обновления. stage: "reinforce" — вес G_t − b, эпизод целиком; "critic" — вместо числа b сеть-критик;
        "tail" — кусок из n шагов, хвост эпизода оценивает критик. Красным выделено то, что изменилось на этом шаге.
    plot_situation(situation, base="b", note=...)
        одна ситуация — четыре действия: кадр и столбики весов. base="b" — вес G − b; base="both" — рядом G − b и G − V(s)
    plot_lander_steps(runs, ax=None, title="", window=100)
        кривые обучения: по горизонтали шаги в среде, по вертикали средняя сумма наград за window эпизодов;
        runs — словарь {вариант: список историй «эпизоды × 2»}
    draw_pieces(length=600, n=128, ax=None)
        когда обновляются сети: после эпизода целиком и после каждого куска из n шагов
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"
ACTIONS = ("ничего", "левый\nдвигатель", "главный\nдвигатель", "правый\nдвигатель")
COLOURS = {"REINFORCE": C_GREY, "актор-критик": C_ACCENT, "критик": C_AGENT, "DQN": C_ENV}      # порядок важен: первое совпадение


def _box(ax, x, y, w, h, text, color, face="#ffffff", size=12, weight="normal", lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def _arrow(ax, a, b, color="k", lw=2.2, rad=0.0, ls="-", scale=20):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def draw_ac_scheme(stage="reinforce", ax=None):
    """Одно обновление стратегии-сети. Три стадии отличаются одним местом каждая — оно выделено красным."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5.6))
    new_critic = C_ACCENT if stage == "critic" else C_REWARD
    new_tail = C_ACCENT if stage == "tail" else "k"
    # среда и стратегия-сеть
    _box(ax, 0.0, 2.6, 2.5, 1.2, "среда:\nлунный модуль", C_ENV, size=13, weight="bold")
    _box(ax, 7.2, 5.3, 3.6, 1.1, "стратегия-сеть (актор)\n$\\pi_\\theta(a \\mid s)$", C_AGENT, face="#dfe7f3", size=13, weight="bold")
    _arrow(ax, (7.1, 5.85), (1.25, 3.95), C_AGENT, rad=0.25, lw=1.8)
    ax.text(3.2, 5.75, "действие — случайно,\nпо вероятностям $\\pi_\\theta$", ha="center", fontsize=11.5, color=C_AGENT)
    # что сыграли
    if stage == "tail":
        _box(ax, 3.6, 2.6, 2.9, 1.2, "кусок из $n$ шагов:\n$s_t, a_t, r_t$", new_tail, face="#f7dfe0", size=12.5, weight="bold")
    else:
        _box(ax, 3.6, 2.6, 2.9, 1.2, "эпизод целиком:\n$s_t, a_t, r_t$", "k", size=12.5)
    _arrow(ax, (2.6, 3.2), (3.5, 3.2), C_ENV)
    # G_t
    if stage == "tail":
        _box(ax, 7.6, 2.6, 3.7, 1.2, "$G_t$ = награды куска\n+ оценка хвоста", new_tail, face="#f7dfe0", size=12.5, weight="bold")
    else:
        _box(ax, 7.6, 2.6, 3.7, 1.2, "$G_t$ — награды с шага $t$\nдо конца эпизода", "k", size=12.5)
    _arrow(ax, (6.6, 3.2), (7.5, 3.2), "k")
    # вес
    base = "b" if stage == "reinforce" else "V_w(s_t)"
    _box(ax, 12.6, 2.6, 3.0, 1.2, f"вес действия\n$G_t - {base}$", "k", size=13, weight="bold")
    _arrow(ax, (11.4, 3.2), (12.5, 3.2), "k")
    _arrow(ax, (14.1, 3.9), (10.9, 5.85), C_AGENT, rad=0.25, lw=2.6)
    ax.text(14.9, 5.45, "шаг градиента:\nдействия с весом > 0\nстановятся вероятнее", ha="center", fontsize=11.5, color=C_AGENT, weight="bold")
    # с чем сравниваем G_t
    if stage == "reinforce":
        _box(ax, 9.2, 0.3, 3.8, 1.1, "$b$ — одно число:\nсредний $G_t$ всех шагов", C_GREY, face="#eeeeee", size=12.5)
        _arrow(ax, (13.1, 0.85), (14.4, 2.5), C_GREY, rad=0.25)
    else:
        face = "#f7dfe0" if stage == "critic" else "#fbe9dc"
        _box(ax, 9.2, 0.3, 3.8, 1.1, "сеть-критик $V_w(s)$:\nсвоя оценка для каждого $s$", new_critic, face=face, size=12.5, weight="bold")
        _arrow(ax, (13.1, 0.85), (14.4, 2.5), new_critic, rad=0.25)
        _arrow(ax, (11.0, 2.5), (11.4, 1.5), new_critic, lw=1.8)
        ax.text(11.5, 2.0, "учится по $G_t$", ha="left", va="center", fontsize=11.5, color=new_critic)
    if stage == "tail":
        _arrow(ax, (9.1, 0.85), (8.3, 2.5), new_tail, rad=-0.3, lw=2.6)
        ax.text(7.75, 1.35, "хвост — оценка критика\nдля состояния после куска", ha="right", va="center", fontsize=12,
                color=new_tail, weight="bold")
    ax.set_xlim(-0.3, 16.6); ax.set_ylim(-0.1, 6.8); ax.axis("off")
    return ax


def _weight_bars(ax, G_actions, base, title, base_label):
    weights = np.asarray(G_actions) - base
    colours = [C_ENV if w > 0 else C_ACCENT for w in weights]
    ax.bar(range(4), weights, color=colours, width=0.62)
    span = np.abs(weights).max()
    for k, w in enumerate(weights):
        ax.text(k, w + (0.04 if w > 0 else -0.04) * span, f"{w:+.0f}".replace("-", "−"), ha="center", va="bottom" if w > 0 else "top",
                fontsize=14, weight="bold")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(range(4)); ax.set_xticklabels(ACTIONS, fontsize=12)
    ax.set_ylim(min(weights.min(), 0) - 0.22 * span, max(weights.max(), 0) + 0.22 * span)
    ax.set_ylabel("вес действия", fontsize=12.5); ax.tick_params(axis="y", labelsize=11)
    ax.set_title(f"{title}\n{base_label}", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_situation(situation, base="b", note="модуль у самой поверхности"):
    """Одна ситуация — четыре действия. Слева кадр, справа веса действий: G − b и (при base="both") G − V(s)."""
    G, b, V = situation["G"], float(situation["b"]), float(situation["V"])
    panels = 3 if base == "both" else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.4 * panels + 0.6, 4.4),
                             gridspec_kw=dict(width_ratios=[1.15] + [1] * (panels - 1)))
    axes[0].imshow(situation["frame"]); axes[0].axis("off")
    axes[0].set_title("ситуация: " + note, fontsize=13)
    _weight_bars(axes[1], G, b, "вес $G - b$", f"$b = {b:.0f}$ — одно число на все ситуации")
    if base == "both":
        _weight_bars(axes[2], G, V, "вес $G - V(s)$", f"$V(s) = {V:.0f}$ — средний $G$ из этой ситуации")
    fig.tight_layout()
    return axes


def plot_lander_steps(runs, ax=None, title="", window=100, limit=None):
    """Кривые обучения по шагам в среде. Тонкие линии — запуски, жирная — среднее по запускам."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.6))
    for name, histories in runs.items():
        colour = next((c for key, c in COLOURS.items() if key in name), C_AGENT)
        end = min(history[-1, 0] for history in histories) if limit is None else limit
        grid = np.linspace(0, end, 200)
        curves = []
        for history in histories:
            steps, totals = np.asarray(history).T
            smooth = np.array([totals[max(0, k - window + 1):k + 1].mean() for k in range(len(totals))])
            curves.append(np.interp(grid, steps, smooth))
            ax.plot(grid / 1000, curves[-1], color=colour, lw=1, alpha=0.28)
        ax.plot(grid / 1000, np.mean(curves, axis=0), color=colour, lw=3.2, label=name)
    ax.axhline(200, color=C_GREY, lw=1.5, ls="--")
    ax.text(0, 208, "200 и выше — модуль сел", ha="left", va="bottom", fontsize=11.5, color=C_GREY)
    ax.set_ylim(-250, 300)
    ax.set_xlabel("шагов в среде, тысяч", fontsize=12.5)
    ax.set_ylabel(f"сумма наград за эпизод,\nсреднее за {window} эпизодов", fontsize=12.5)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=12.5)
    ax.set_title(title, fontsize=13)
    return ax


def draw_pieces(length=600, n=128, ax=None):
    """Один эпизод из length шагов: когда обновляются сети, если ждать конца эпизода и если резать на куски по n шагов."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.9))
    # верхняя полоса: эпизод целиком
    ax.add_patch(Rectangle((0, 2.2), length, 0.7, facecolor="#e9e9e9", edgecolor="k", lw=1.5))
    ax.text(length / 2, 2.55, f"эпизод целиком: {length} шагов", ha="center", va="center", fontsize=12.5)
    ax.plot([length], [2.55], marker="v", ms=15, color=C_AGENT, clip_on=False)
    ax.text(length + 12, 2.55, "1 обновление", ha="left", va="center", fontsize=12.5, color=C_AGENT, weight="bold")
    ax.text(-12, 2.55, "ждём конца\nэпизода", ha="right", va="center", fontsize=12.5)
    # нижняя полоса: куски
    edges = list(range(0, length, n)) + [length]
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ax.add_patch(Rectangle((a, 0.6), b - a, 0.7, facecolor="#f7dfe0" if k % 2 == 0 else "#fbe9dc", edgecolor="k", lw=1.5))
        ax.text((a + b) / 2, 0.95, f"{b - a}", ha="center", va="center", fontsize=12)
        ax.plot([b], [0.95], marker="v", ms=15, color=C_ACCENT, clip_on=False)
    ax.text(length + 12, 0.95, f"{len(edges) - 1} обновлений", ha="left", va="center", fontsize=12.5, color=C_ACCENT, weight="bold")
    ax.text(-12, 0.95, f"куски по\n{n} шагов", ha="right", va="center", fontsize=12.5)
    # хвост первого куска
    ax.annotate("", xy=(length, 0.25), xytext=(n, 0.25), arrowprops=dict(arrowstyle="-|>", color=C_REWARD, lw=2.2, ls=(0, (5, 3))))
    ax.text((n + length) / 2, -0.02, "хвост первого куска ещё не сыгран — вместо него оценка критика $V(s)$ для состояния после куска",
            ha="center", va="top", fontsize=12, color=C_REWARD)
    ax.text(length + 12, 0.3, "в конце эпизода хвоста нет", ha="left", va="center", fontsize=11, color=C_GREY)
    ax.set_xlim(-95, length + 150); ax.set_ylim(-0.6, 3.2); ax.axis("off")
    return ax
