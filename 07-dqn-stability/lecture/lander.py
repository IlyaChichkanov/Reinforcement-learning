"""Тот же DQN, что `train_dqn` лекции 7, для посадки лунного модуля (среда LunarLander-v3 из Gymnasium).

Отличий от шеста два: размеры сети (8 чисел состояния на входе, 4 действия на выходе) и сама среда.
Буфер опыта, цель от копии сети и сдвиг копии на 1 % за шаг — те же.

    make_lander(render=False)         среда; render=True — умеет отдавать кадры (env.render() -> картинка)
    make_lander_net(seed=0)           Q-сеть 8 → 64 → 64 → 4
    greedy_action(qnet, s)            действие с наибольшим Q
    train_lander(seed, episodes=600, lr=1e-3, batch=64, tau=0.01, eps_steps=20_000)
        возвращает (суммы наград по эпизодам, сеть)
    play_lander(qnet, seed=0)         один эпизод без случайных действий: (кадры, сумма наград)

Состояние: x, y, скорость по x и по y, угол, угловая скорость, касание левой и правой опоры (0 или 1).
Действия: 0 — ничего, 1 — левый двигатель, 2 — главный, 3 — правый.
"""
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from dqn_core import q_of, td_targets, to_tensors


def make_lander(render=False):
    return gym.make("LunarLander-v3", render_mode="rgb_array" if render else None)


def make_lander_net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 4))


def greedy_action(qnet, s):
    with torch.no_grad():
        return int(qnet(torch.as_tensor(s)).argmax())


def train_lander(seed, episodes=600, lr=1e-3, batch=64, tau=0.01, eps_steps=20_000, log=None):
    rng, env, qnet = np.random.default_rng(seed), make_lander(), make_lander_net(seed)
    target_net = make_lander_net(seed)                                 # копия сети, веса θ̄
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = deque(maxlen=50_000)                                      # журнал переходов
    step, returns = 0, []
    for episode in range(episodes):
        s, _ = env.reset(seed=seed * 10_000 + episode)
        total, done = 0.0, False
        while not done:
            # 1. шаг в среде, переход — в журнал; ε падает от 1 до 0.01 за eps_steps шагов
            eps = max(0.01, 1 - step / eps_steps)
            a = int(rng.integers(4)) if rng.random() < eps else greedy_action(qnet, s)
            s2, r, terminated, truncated, _ = env.step(a)
            buffer.append((s, a, np.float32(r), s2, np.float32(terminated)))
            # 2. пачка из 64 случайных переходов
            S, A, R, S2, D = to_tensors([buffer[i] for i in rng.integers(len(buffer), size=batch)])
            # 3. цель считает копия; шаг градиента
            y = td_targets(target_net, R, S2, D)
            loss = ((y - q_of(qnet, S, A)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            # 4. копия подтягивается к основной сети на долю τ
            with torch.no_grad():
                for w_copy, w in zip(target_net.parameters(), qnet.parameters()):
                    w_copy.mul_(1 - tau).add_(tau * w)
            step += 1; s = s2; total += r
            done = terminated or truncated
        returns.append(total)
        if log and (episode + 1) % 25 == 0:
            log(episode + 1, step, float(np.mean(returns[-25:])))
    return np.array(returns), qnet


def play_lander(qnet, seed=0):
    env = make_lander(render=True)
    s, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        a = greedy_action(qnet, s) if qnet is not None else int(env.action_space.sample())
        s, r, terminated, truncated, _ = env.step(a)
        frames.append(env.render()); total += r
        done = terminated or truncated
    return frames, total
