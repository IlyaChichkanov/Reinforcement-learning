"""Служебный код лекции 8: лунный модуль и стратегия-сеть для него.

Это код лекции 5 (стратегия-сеть, эпизод, G_t, log π), переписанный с шеста на лунный модуль:
у модуля 8 чисел состояния и 4 действия. Три функции обучения — в ячейках лекции (и те же — в ac_train.py).

    make_lander(render=False)             среда LunarLander-v3; render=True — умеет отдавать кадры
    make_net(n_out, seed)                 сеть 8 → 64 → 64 → n_out: n_out = 4 — стратегия (актор), n_out = 1 — критик
    choose_action(actor, s, rng)          действие по вероятностям стратегии-сети π(· | s)
    play(actor, env, s, rng, n)           до n шагов стратегией-сетью из состояния s (раньше — если эпизод закончился);
                                          возвращает (состояния, действия, награды, следующее состояние, terminated, truncated)
    returns_to_go(rewards, gamma=0.99, tail=0.0)
                                          G_t, посчитанные с конца куска; tail — хвост: оценка того, что будет после куска
    log_probs_of(actor, states, actions)  log π(a_t | s_t) для всех шагов — тензор по шагам

Сохранённые результаты (lander_ac_results.npz) — чтобы ноутбук не обучал модуль заново при каждом запуске:

    load_runs(kind)                       истории обучения по запускам: kind — "reinforce", "critic", "ac" или "dqn" (лекция 7, 3 запуска);
                                          история — массив «эпизоды × 2»: (шагов в среде сделано, сумма наград эпизода)
    load_episode()                        награды одного настоящего эпизода с посадкой (для картинки G_t)
    load_situation()                      опыт «одно состояние — четыре действия»: словарь с числами (см. measure_actions)

Обвязка опытов:

    print_runs(name, histories)           сводка варианта по запускам, три строки: среднее последних 100 эпизодов;
                                          в скольких запусках скользящее среднее дошло до 200 и за сколько шагов;
                                          доля последних эпизодов, кончившихся по времени
    steps_to_land(history, level=200, window=100)
                                          сколько шагов в среде сделано, когда среднее window эпизодов впервые достигло level
    measure_actions(actor, seed, t, repeats=300)
                                          одно состояние: средний G после каждого из четырёх действий, вероятности действий
                                          и V(s) — средний G, если из этого состояния играет сама стратегия
    play_frames(actor, seed)              один эпизод стратегией-сетью: (кадры, сумма наград, вышло ли время)
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ACTIONS = ("ничего", "левый\nдвигатель", "главный\nдвигатель", "правый\nдвигатель")
RESULTS = Path(__file__).with_name("lander_ac_results.npz")


def make_lander(render=False):
    return gym.make("LunarLander-v3", render_mode="rgb_array" if render else None)


def make_net(n_out, seed):
    # сеть для лунного модуля: состояние (8 чисел) -> n_out чисел
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, n_out))


def choose_action(actor, s, rng):
    # действие по вероятностям стратегии-сети π(· | s)
    with torch.no_grad():
        probs = torch.softmax(actor(torch.as_tensor(s)), dim=-1).numpy().astype(float)
    return int(rng.choice(4, p=probs / probs.sum()))


def play(actor, env, s, rng, n):
    # до n шагов стратегией-сетью из состояния s; раньше — если эпизод закончился
    states, actions, rewards = [], [], []
    for _ in range(n):
        a = choose_action(actor, s, rng)
        s2, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r); s = s2
        if terminated or truncated:
            break
    return np.array(states), np.array(actions), np.array(rewards), s, terminated, truncated


def returns_to_go(rewards, gamma=0.99, tail=0.0):
    # G_t — награды с шага t до конца куска плюс tail — оценка того, что будет после него
    G, acc = np.zeros(len(rewards)), tail
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc; G[t] = acc
    return G


def log_probs_of(actor, states, actions):
    # log π(a_t | s_t) для всех шагов: из строки t таблицы «шаги × 4 действия» — столбец сыгранного действия a_t
    log_pi = torch.log_softmax(actor(torch.as_tensor(states)), dim=-1)
    return log_pi[torch.arange(len(actions)), torch.as_tensor(actions)]


# ------------------------------------------------------------------ сохранённые результаты

def load_runs(kind):
    data = np.load(RESULTS)
    return [data[name].astype(float) for name in sorted(k for k in data.files if k.startswith(f"{kind}_history_"))]


def load_episode():
    """Награды одного настоящего эпизода с посадкой — для напоминания, что такое G_t."""
    return np.load(RESULTS)["episode_rewards"].astype(float)


def load_situation():
    data = np.load(RESULTS)
    return {key[len("situation_"):]: data[key] for key in data.files if key.startswith("situation_")}


# ------------------------------------------------------------------ обвязка опытов

def steps_to_land(history, level=200, window=100):
    steps, totals = np.asarray(history).T
    for k in range(window, len(totals) + 1):
        if totals[k - window:k].mean() >= level:
            return int(steps[k - 1])
    return None


def print_runs(name, histories):
    """Сводка одного варианта обучения по запускам; histories — список историй. Печатает три строки."""
    last = np.array([history[-100:, 1].mean() for history in histories])
    landed = sorted(round(steps / 1000) for steps in map(steps_to_land, histories) if steps is not None)
    timeouts = sorted(round(100 * (np.diff(history[:, 0], prepend=0)[-100:] >= 1000).mean()) for history in histories)
    print(name)
    print(f"    среднее последних 100 эпизодов: {last.mean():.0f}; по запускам: {' '.join(f'{x:.0f}' for x in np.sort(last))};"
          f" не меньше 200 — в {(last >= 200).sum()} из {len(last)}")
    print(f"    до среднего 200 дошли {len(landed)} из {len(last)} запусков"
          + (f", шагов в среде, тыс.: {' '.join(map(str, landed))}" if landed else ""))
    print(f"    время вышло (1000 шагов) в последних 100 эпизодах, % по запускам: {' '.join(map(str, timeouts))}")


def measure_actions(actor, seed, t, repeats=300):
    """Одно и то же состояние (эпизод seed после t шагов), четыре действия.
    Для каждого действия: делаем его, дальше играем стратегией-сетью до конца эпизода, считаем G; так repeats раз.
    Ещё repeats раз доигрываем эпизод из того же состояния самой стратегией — это измерение V(s).
    Возвращает (средний G по действиям — 4 числа, вероятности действий — 4 числа, V(s) — число)."""
    env, rng = make_lander(), np.random.default_rng(seed)
    s, _ = env.reset(seed=seed)
    prefix = []
    for _ in range(t):                                                   # как модуль попал в это состояние
        a = choose_action(actor, s, rng)
        s, *_ = env.step(a); prefix.append(a)
    with torch.no_grad():
        probs = torch.softmax(actor(torch.as_tensor(s)), dim=-1).numpy()
    means = []
    for first in range(4):
        totals = []
        for _ in range(repeats):
            env.reset(seed=seed)
            for a in prefix:
                s, *_ = env.step(a)
            s, r, terminated, truncated, _ = env.step(first)
            rewards = [r]
            if not (terminated or truncated):
                rewards += list(play(actor, env, s, rng, n=1000)[2])
            totals.append(returns_to_go(np.array(rewards))[0])
        means.append(np.mean(totals))
    totals = []
    for _ in range(repeats):                                             # V(s): первое действие тоже выбирает стратегия
        env.reset(seed=seed)
        for a in prefix:
            s, *_ = env.step(a)
        totals.append(returns_to_go(play(actor, env, s, rng, n=1000)[2])[0])
    return np.array(means), probs, float(np.mean(totals))


def play_frames(actor, seed):
    env, rng = make_lander(render=True), np.random.default_rng(seed)
    s, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        s, r, terminated, truncated, _ = env.step(choose_action(actor, s, rng))
        frames.append(env.render()); total += r
        done = terminated or truncated
    return frames, total, truncated
