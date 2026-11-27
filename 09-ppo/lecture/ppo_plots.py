"""Картинки лекции 9 «PPO».

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.
Рисовальщики принимают ax=None, рисуют и возвращают ax.

    plot_rho(rhos, eps=0.2, show=300)
        насколько изменилась стратегия: по строке на вариант, точка — шаг куска; красные точки вышли за коридор;
        rhos — словарь {подпись: массив ρ}
    draw_clip(eps=0.2, adv=(+1, -1))
        что видит градиент: слагаемое как функция ρ при положительном и при отрицательном преимуществе, две панели
    plot_lander_steps(runs, ax=None, title="", window=100)
        кривые обучения по шагам в среде; runs — словарь {вариант: список историй «эпизоды × 2»}
    draw_ppo_scheme(ax=None)
        одно обновление PPO: кусок → преимущества → несколько проходов с обрезанным отношением
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_AGENT, C_ENV, C_REWARD, C_ACCENT, C_GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"
COLOURS = {"PPO": C_AGENT, "один проход": C_GREY, "10 проходов": C_ACCENT,      # порядок важен: первое совпадение
           "актор-критик": C_GREY, "непрерывные": C_ENV}


def _colour(name):
    return next((c for key, c in COLOURS.items() if key in name), C_AGENT)


def _box(ax, x, y, w, h, text, color, face="#ffffff", size=12, weight="normal", lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def _arrow(ax, a, b, color="k", lw=2.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def plot_rho(rhos, eps=0.2, show=300, seed=0):
    """Отношения ρ = π_новая / π_старая по шагам куска: по строке на вариант, точка — один шаг.
    Серая полоса — коридор [1 − ε, 1 + ε]; show — сколько шагов показать (берутся случайно)."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(12, 1.3 + 1.1 * len(rhos)))
    ax.axvspan(1 - eps, 1 + eps, color=C_GREY, alpha=0.18)
    ax.axvline(1, color="k", lw=1.2, ls=":")
    names = list(rhos)
    for row, name in enumerate(names):
        rho = np.asarray(rhos[name], dtype=float)
        share = 100 * np.mean((rho < 1 - eps) | (rho > 1 + eps))
        sample = rng.choice(rho, size=min(show, len(rho)), replace=False)
        y = row + rng.uniform(-0.22, 0.22, size=len(sample))
        inside = (sample >= 1 - eps) & (sample <= 1 + eps)
        ax.scatter(np.clip(sample[inside], 0, 3), y[inside], s=14, color=C_AGENT, alpha=0.45)      # в коридоре
        ax.scatter(np.clip(sample[~inside], 0, 3), y[~inside], s=18, color=C_ACCENT, alpha=0.75)   # за коридором
        ax.text(3.05, row, f"за коридором {share:.0f} %", va="center", fontsize=12.5, weight="bold",
                color=C_ACCENT if share > 20 else "k")
    ax.text(1, -0.5, f"коридор [{1 - eps:.1f}, {1 + eps:.1f}]", ha="center", va="center", fontsize=12)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=12.5)
    ax.set_ylim(-0.6, len(names) - 0.3); ax.set_xlim(0, 3)
    ax.set_xlabel(r"$\rho$ — во сколько раз изменилась вероятность сыгранного действия", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return ax


def draw_clip(eps=0.2, adv=(1.0, -1.0)):
    """Что видит градиент: слагаемое min(ρA, clip(ρ)A) как функция ρ — при A > 0 и при A < 0."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), sharey=False)
    grid = np.linspace(0, 2.2, 400)
    for ax, A in zip(axes, adv):
        plain = grid * A
        clipped = np.clip(grid, 1 - eps, 1 + eps) * A
        taken = np.minimum(plain, clipped)
        ax.plot(grid, plain, color=C_GREY, lw=1.8, ls="--", label=r"без клиппинга: $\rho A$")
        ax.plot(grid, taken, color=C_AGENT, lw=3, label=r"PPO: $\min(\rho A,\ \mathrm{clip}(\rho) A)$")
        ax.axvspan(1 - eps, 1 + eps, color=C_GREY, alpha=0.18)
        ax.axvline(1, color="k", lw=1.2, ls=":")
        flat = (1 + eps, 2.2) if A > 0 else (0, 1 - eps)
        ax.axvspan(*flat, color=C_ACCENT, alpha=0.10)
        ax.text(np.mean(flat), A * (0.35 if A > 0 else 0.75), "здесь линия ровная:\nградиента нет",
                ha="center", va="center", fontsize=12, color=C_ACCENT, weight="bold")
        ax.set_title(("действие оказалось лучше обычного: $A > 0$" if A > 0 else
                      "действие оказалось хуже обычного: $A < 0$"), fontsize=13)
        ax.set_xlabel(r"$\rho$", fontsize=13); ax.set_ylabel("слагаемое loss (чем выше, тем лучше)", fontsize=12)
        ax.tick_params(labelsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=11.5, loc="lower right" if A > 0 else "upper right")
    fig.tight_layout()
    return axes


def plot_lander_steps(runs, ax=None, title="", window=100):
    """Кривые обучения по шагам в среде: тонкие линии — запуски, жирная — среднее по запускам."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.6))
    for name, histories in runs.items():
        colour = _colour(name)
        end = min(history[-1, 0] for history in histories)
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
    ax.set_ylim(-300, 300)
    ax.set_xlabel("шагов в среде, тысяч", fontsize=12.5)
    ax.set_ylabel(f"сумма наград за эпизод,\nсреднее за {window} эпизодов", fontsize=12.5)
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=12)
    ax.set_title(title, fontsize=13)
    return ax


def draw_ppo_scheme(ax=None):
    """Одно обновление PPO: кусок играется один раз, а проходов по нему несколько."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4.6))
    _box(ax, 0.0, 2.2, 2.6, 1.1, "среда:\nлунный модуль", C_ENV, size=12.5, weight="bold")
    _box(ax, 3.4, 2.2, 3.4, 1.1, "кусок из $n$ шагов:\n$s_t, a_t, r_t$ и $\\log \\pi_{\\text{старая}}$", C_ACCENT,
         face="#f7dfe0", size=12.5, weight="bold")
    _arrow(ax, (2.7, 2.75), (3.3, 2.75), C_ENV)
    _box(ax, 7.6, 2.2, 3.0, 1.1, "преимущества\n$A_t = G_t - V_w(s_t)$", "k", size=12.5)
    _arrow(ax, (6.9, 2.75), (7.5, 2.75), "k")
    _box(ax, 11.4, 2.2, 3.2, 1.1, "$\\rho_t = \\pi_\\theta / \\pi_{\\text{старая}}$,\nобрезаем и шагаем", C_AGENT,
         face="#dfe7f3", size=12.5, weight="bold")
    _arrow(ax, (10.7, 2.75), (11.3, 2.75), "k")
    _arrow(ax, (13.0, 2.1), (13.0, 1.2), C_AGENT)
    _arrow(ax, (12.4, 1.2), (12.4, 2.1), C_AGENT)
    ax.text(13.2, 1.65, "так 10 раз по одному куску", ha="left", va="center", fontsize=12.5, color=C_AGENT, weight="bold")
    _arrow(ax, (11.4, 3.5), (1.3, 3.5), C_AGENT, rad=0.12, lw=1.8)
    ax.text(6.4, 4.0, "после всех проходов стратегия идёт играть следующий кусок", ha="center", fontsize=12, color=C_AGENT)
    ax.text(5.1, 1.5, "кусок сыгран один раз — шаги в среде тратятся только здесь", ha="center", fontsize=12, color=C_ACCENT)
    ax.set_xlim(-0.3, 17.4); ax.set_ylim(0.8, 4.4); ax.axis("off")
    return ax
