"""Анимации (GIF) для семинара 1: скольжение, эпизод, отбор элиты, обучение Cross-Entropy.

Запуск из корня репозитория:

    python assets/make_animations.py              # все
    python assets/make_animations.py slip episode # только названные

GIF пишутся в ``assets/anim/`` и встраиваются в ноутбук вложениями
(``tools/embed_media.py``), поэтому видны на GitHub и в Colab без запуска ноутбука.
mp4 сознательно не используем: в окружении нет ffmpeg, доступны только writer'ы
``pillow`` и ``html``.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01-intro" / "seminar"))
from lake_plots import (ARROWS, ACTION_NAMES, C_ACCENT, C_AGENT, C_ENV, C_GREY,
                        draw_lake, draw_policy, lake_desc, plot_returns, show_transitions)

OUT = ROOT / "assets" / "anim"
DPI = 90
GAMMA = 0.95


def _save(ani, name, fps):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gif"
    ani.save(path, writer=PillowWriter(fps=fps), dpi=DPI)
    print(f"saved {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


# --------------------------------------------------------------------------- среда и CEM
def _run(env, policy, rng, max_steps=100):
    s, _ = env.reset(seed=int(rng.integers(1_000_000)))
    states, actions, rewards = [s], [], []
    for _ in range(max_steps):
        a = int(rng.choice(len(policy[s]), p=policy[s]))
        s, r, term, trunc, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        if term or trunc:
            break
    return states, actions, rewards


def _ret(rewards, gamma=GAMMA):
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def _elite(sessions, returns, q):
    thr = np.quantile(returns, q)
    mask = np.asarray(returns) >= max(thr, np.min(returns) + 1e-12)
    return thr, mask


def _update(sessions, mask, shape, laplace=0.5, mix=0.5, old=None):
    counts = np.full(shape, laplace)
    for take, (states, actions, _) in zip(mask, sessions):
        if take:
            for s, a in zip(states, actions):
                counts[s, a] += 1
    new = counts / counts.sum(axis=1, keepdims=True)
    return new if old is None else (1 - mix) * old + mix * new


# --------------------------------------------------------------------------- 1. скольжение
def slip():
    """Просим «вправо» из клетки 6 — и видим, что лёд решает иначе. Частоты сходятся к 1/3."""
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env.reset(seed=0)
    desc, s, a = lake_desc(env), 6, 2
    outcomes = sorted({sn for _, sn, _, _ in env.unwrapped.P[s][a]})
    rng = np.random.default_rng(0)
    draws = [int(rng.choice(outcomes)) for _ in range(60)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.3))

    def frame(k):
        axL.clear(); axR.clear()
        show_transitions(env, s, a, ax=axL, title=f"просим «{ACTION_NAMES[a]}» из клетки {s}")
        if k > 0:
            r1, c1 = divmod(draws[k - 1], len(desc[0]))
            axL.add_patch(Circle((c1 + 0.5, r1 + 0.5), 0.17, color=C_ACCENT, zorder=5))
        seen = draws[:k]
        freq = [seen.count(o) / max(k, 1) for o in outcomes]
        labels = [f"клетка {o}" + ("\n(прорубь)" if desc[o // 4][o % 4] == "H" else "") for o in outcomes]
        axR.bar(labels, freq, color=[C_ACCENT if desc[o // 4][o % 4] == "H" else C_ENV for o in outcomes])
        axR.axhline(1 / 3, ls="--", lw=1.2, color=C_GREY)
        axR.text(2.45, 1 / 3 + 0.02, "1/3", fontsize=8, color=C_GREY, ha="right")
        axR.set_ylim(0, 0.75); axR.set_ylabel("доля попыток"); axR.tick_params(labelsize=8)
        axR.set_title(f"попыток: {k}", fontsize=10)

    ani = FuncAnimation(fig, frame, frames=len(draws) + 1, interval=160)
    _save(ani, "slip", fps=6); plt.close(fig)


# --------------------------------------------------------------------------- 2. эпизод
PLAN = [1, 1, 2, 1, 2, 2]            # ↓ ↓ → ↓ → → : маршрут 0-4-8-9-13-14-15 по твёрдому льду
_DR, _DC = [0, 1, 0, -1], [-1, 0, 1, 0]


def _intended(s, a, ncol=4):
    r, c = divmod(s, ncol); rr, cc = r + _DR[a], c + _DC[a]
    return rr * ncol + cc if 0 <= rr < ncol and 0 <= cc < ncol else s


def _play(env, plan, seed):
    s, _ = env.reset(seed=seed)
    states, acts, rews = [s], [], []
    for a in plan:
        s, r, term, trunc, _ = env.step(a)
        states.append(s); acts.append(a); rews.append(r)
        if term or trunc:
            break
    return states, acts, rews


def episode():
    """Один и тот же план на твёрдом и на скользком льду: намерение против результата."""
    hard = gym.make("FrozenLake-v1", is_slippery=False)
    slick = gym.make("FrozenLake-v1", is_slippery=True)
    hard.reset(seed=0); desc, ncol = lake_desc(hard), 4

    runs = [("твёрдый лёд: план работает", _play(hard, PLAN, 0), C_ENV)]
    for seed in range(200):                              # берём прогон, где лёд явно вмешался
        st, ac, rw = _play(slick, PLAN, seed)
        if sum(1 for i, a in enumerate(ac) if st[i + 1] != _intended(st[i], a)) >= 2:
            runs.append(("скользкий лёд: тот же план", (st, ac, rw), C_ACCENT))
            break

    n = max(len(st) for _, (st, _, _), _ in runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3))

    def frame(t):
        for ax, (head, (states, actions, rewards), col) in zip(axes, runs):
            ax.clear(); draw_lake(desc, ax=ax, numbers=True)
            k = min(t, len(states) - 1)
            for i in range(k):
                r0, c0 = divmod(states[i], ncol); r1, c1 = divmod(states[i + 1], ncol)
                ax.plot([c0 + .5, c1 + .5], [r0 + .5, r1 + .5], color=C_AGENT, alpha=.35, lw=2, zorder=2)
            r, c = divmod(states[k], ncol)
            ax.add_patch(Circle((c + .5, r + .5), .2, color=C_AGENT, zorder=3))
            if k < len(actions):
                mark = "как просили" if states[k + 1] == _intended(states[k], actions[k]) else "лёд понёс не туда"
                ax.set_title(f"{head}\nшаг {k}: просим «{ARROWS[actions[k]]}» — {mark}", fontsize=9)
            else:
                win = desc[states[-1] // ncol][states[-1] % ncol] == "G"
                end = "дошли до цели" if win else "провалились в прорубь"
                ax.set_title(f"{head}\n{end}, return = {_ret(rewards):.2f}", fontsize=9, color=col)

    ani = FuncAnimation(fig, frame, frames=n + 1, interval=900)
    _save(ani, "episode", fps=1.1); plt.close(fig)


# --------------------------------------------------------------------------- CEM-прогон
def _cem_log(slippery=False, n_iter=14, n_sessions=250, q=0.7, seed=0):
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    rng = np.random.default_rng(seed)
    policy = np.ones((16, 4)) / 4
    log = {"policies": [policy], "returns": [], "thr": [], "mask": [], "success": []}
    for _ in range(n_iter):
        sessions = [_run(env, policy, rng) for _ in range(n_sessions)]
        returns = np.array([_ret(r) for *_, r in sessions])
        thr, mask = _elite(sessions, returns, q)
        log["returns"].append(returns); log["thr"].append(thr); log["mask"].append(mask)
        log["success"].append(float(np.mean([r[-1] == 1.0 if r else False for *_, r in sessions])))
        policy = _update(sessions, mask, policy.shape, old=policy)
        log["policies"].append(policy)
    return lake_desc(env), log


# --------------------------------------------------------------------------- 3. отбор элиты
def elite():
    """Что делает select_elite: облако эпизодов, порог-квантиль и выжившая элита."""
    desc, log = _cem_log()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    def frame(k):
        ax.clear()
        plot_returns(log["returns"][k], threshold=log["thr"][k], elite_mask=log["mask"][k], ax=ax,
                     title=f"итерация {k}: обучаемся на {int(log['mask'][k].sum())} лучших из {len(log['returns'][k])}")

    ani = FuncAnimation(fig, frame, frames=len(log["returns"]), interval=900)
    _save(ani, "elite", fps=1.1); plt.close(fig)


# --------------------------------------------------------------------------- 4. обучение CEM
def cem():
    """Политика по итерациям: из «кубика» вырастают стрелки, доля успехов ползёт вверх."""
    desc, log = _cem_log()
    fig, ax = plt.subplots(figsize=(3.6, 4.0))

    def frame(k):
        ax.clear()
        draw_policy(log["policies"][k], desc, ax=ax,
                    title=f"итерация {k}\nдоля успехов {log['success'][min(k, len(log['success']) - 1)]:.0%}")

    ani = FuncAnimation(fig, frame, frames=len(log["policies"]), interval=900)
    _save(ani, "cem", fps=1.1); plt.close(fig)


ALL = {"slip": slip, "episode": episode, "elite": elite, "cem": cem}


# ═══════════════════════════════════════════════════════════ семинар 2: клетчатый лабиринт
sys.path.insert(0, str(ROOT / "02-environments" / "lecture"))
from gridworld import (GridWorld, mdp_matrices, uniform_policy, greedy_policy,   # noqa: E402
                       run_episode, estimate_values_mc, solve_q_star)
from gridworld_plots import draw_policy, plot_values                             # noqa: E402

MAZE = ["....G", ".##.X", "S...."]            # тот же лабиринт, что в семинаре 2


def _maze(noise=0.1, step_reward=-0.04):
    env = GridWorld(layout=MAZE, step_reward=step_reward, noise=noise)
    P, R = mdp_matrices(env)
    return env, P, R


def vi_wave():
    """Волна ценности: знание о выходе расходится от него по одной клетке за итерацию."""
    env, P, R = _maze()
    hist, Q = [np.zeros_like(R)], np.zeros_like(R)
    for _ in range(22):
        Q = R + GAMMA * (P @ Q.max(axis=1))
        hist.append(Q)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    def frame(k):
        ax.clear()
        plot_values(hist[k].max(axis=1), env, ax=ax,
                    title="начальное приближение: одни нули" if k == 0 else f"после {k} итераций")

    _save(FuncAnimation(fig, frame, frames=len(hist), interval=700), "vi_wave", fps=1.4)
    plt.close(fig)


def eval_vs_mc():
    """Монте-Карло шумит и сходится медленно, уравнение даёт тот же ответ сразу и точно."""
    env, P, R = _maze()
    policy = uniform_policy(env)
    Q = np.zeros_like(R)
    for _ in range(400):
        Q = R + GAMMA * (P @ (policy * Q).sum(axis=1))
    V_exact = (policy * Q).sum(axis=1)

    rng = np.random.default_rng(1)
    sessions = [run_episode(env, policy, rng) for _ in range(2000)]
    counts = [5, 10, 20, 40, 80, 150, 300, 600, 1000, 1500, 2000]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

    def frame(k):
        n = counts[k]
        V_mc = estimate_values_mc(sessions[:n], env.n_states)
        diff = np.nanmax(np.abs(V_mc - V_exact))
        axes[0].clear(); axes[1].clear()
        plot_values(V_mc, env, ax=axes[0], title=f"Монте-Карло: {n} эпизодов")
        plot_values(V_exact, env, ax=axes[1], title="решение уравнения")
        axes[0].set_xlabel(f"расхождение {diff:.2f}", fontsize=9, color=C_ACCENT, labelpad=6)

    _save(FuncAnimation(fig, frame, frames=len(counts), interval=900), "eval_vs_mc", fps=1.1)
    plt.close(fig)


def improve_loop():
    """Оценили — улучшили, и так по кругу: стрелки поворачиваются, ценность растёт и не падает нигде."""
    env, P, R = _maze()

    def evaluate(policy, n_iter=400):
        Q = np.zeros_like(R)
        for _ in range(n_iter):
            Q = R + GAMMA * (P @ (policy * Q).sum(axis=1))
        return Q, (policy * Q).sum(axis=1)

    policy = uniform_policy(env)
    steps = []
    for _ in range(5):
        Q_pi, V_pi = evaluate(policy)
        steps.append((policy, V_pi))
        nxt = greedy_policy(Q_pi)
        if np.array_equal(nxt, policy):
            break
        policy = nxt

    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    def frame(k):
        policy, V = steps[k]
        ax.clear()
        step_word = "шаг" if k == 1 else "шага" if k in (2, 3, 4) else "шагов"
        head = "старт: случайная стратегия" if k == 0 else f"после {k} {step_word} улучшения"
        draw_policy(policy, env, ax=ax, values=V, title=f"{head}\nценность старта {V[env.start]:+.2f}")

    _save(FuncAnimation(fig, frame, frames=len(steps), interval=1300), "improve_loop", fps=0.8)
    plt.close(fig)


def wind_sweep():
    """Чем сильнее ветер, тем дальше оптимальная стратегия уводит агента от ямы."""
    noises = np.round(np.arange(0.0, 0.42, 0.04), 2)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    def frame(k):
        env, P, R = _maze(noise=float(noises[k]))
        V = solve_q_star(P, R).max(axis=1)
        ax.clear()
        draw_policy(greedy_policy(solve_q_star(P, R)), env, ax=ax, values=V,
                    title=f"ветер {noises[k]:.2f} — сносит вбок с такой вероятностью\n"
                          f"ценность старта {V[env.start]:+.2f}")

    _save(FuncAnimation(fig, frame, frames=len(noises), interval=1100), "wind_sweep", fps=0.9)
    plt.close(fig)


ALL.update({"vi_wave": vi_wave, "eval_vs_mc": eval_vs_mc,
            "improve_loop": improve_loop, "wind_sweep": wind_sweep})


if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        if n not in ALL:
            raise SystemExit(f"неизвестная анимация: {n}. Доступны: {', '.join(ALL)}")
        ALL[n]()
