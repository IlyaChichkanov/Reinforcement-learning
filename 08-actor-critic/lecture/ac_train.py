"""Три функции обучения лекции 8 — тот же текст, что в ячейках ноутбука (файл собирается вместе с ноутбуком).

    train_reinforce(seed, steps=STEPS)        REINFORCE: вес G_t − b; возвращает (история, актор)
    train_critic(seed, steps=STEPS)           вес G_t − V(s_t), обновление после эпизода; возвращает (история, актор, критик)
    train_ac(seed, steps=STEPS, n=128)        актор-критик: обновление после куска из n шагов, хвост оценивает критик

История — массив «эпизоды × 2»: (сколько шагов в среде сделано к концу эпизода, сумма наград эпизода).
Нужен этот файл для assets/make_animations.py: по нему посчитаны сохранённые результаты lander_ac_results.npz.
"""
import numpy as np
import torch

from ac_core import log_probs_of, make_lander, make_net, play, returns_to_go

STEPS = 1_000_000          # шагов в среде на один запуск


def train_reinforce(seed, steps=STEPS):
    # REINFORCE лекции 5: вес действия G_t − b, где b — средний G_t всех прошлых шагов
    rng, env, actor = np.random.default_rng(seed), make_lander(), make_net(4, seed)
    opt = torch.optim.Adam(actor.parameters(), lr=0.001)
    b, count, played, history = 0.0, 0, 0, []                             # played — сколько шагов в среде сыграно
    while played < steps:
        s, _ = env.reset(seed=seed * 10_000 + len(history))
        states, actions, rewards, *_ = play(actor, env, s, rng, n=1000)   # эпизод целиком: не больше 1000 шагов
        G = returns_to_go(rewards)                                         # массив: G_t для каждого шага
        weights = torch.as_tensor(G - b, dtype=torch.float32)              # веса действий — числа
        loss = -(weights * log_probs_of(actor, states, actions)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        for g in G:                                                        # b — средний G_t всех прошлых шагов
            count += 1; b += (g - b) / count
        played += len(rewards); history.append((played, rewards.sum()))
    return np.array(history), actor


def train_critic(seed, steps=STEPS):
    # то же, но вместо числа b — сеть-критик V(s)
    rng, env, actor = np.random.default_rng(seed), make_lander(), make_net(4, seed)
    critic = make_net(1, seed + 1000)                                      # ← новое: сеть-критик, на выходе одно число
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.001)
    played, history = 0, []
    while played < steps:
        s, _ = env.reset(seed=seed * 10_000 + len(history))
        states, actions, rewards, *_ = play(actor, env, s, rng, n=1000)
        G = torch.as_tensor(returns_to_go(rewards), dtype=torch.float32)
        V = critic(torch.as_tensor(states)).squeeze(1)                     # ← новое: V(s_t) для каждого шага
        weights = (G - V).detach()                                         # ← новое: вместо G − b; detach — готовые числа
        loss_actor = -(weights * log_probs_of(actor, states, actions)).sum()
        loss_critic = ((G - V) ** 2).sum()                                 # ← новое: учит критика
        loss = loss_actor + loss_critic
        opt.zero_grad(); loss.backward(); opt.step()
        played += len(rewards); history.append((played, rewards.sum()))
    return np.array(history), actor, critic


def train_ac(seed, steps=STEPS, n=128):
    # актор-критик: сети обновляются после каждого куска из n шагов, хвост эпизода оценивает критик
    rng, env, actor, critic = np.random.default_rng(seed), make_lander(), make_net(4, seed), make_net(1, seed + 1000)
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": 0.0007},
                            {"params": critic.parameters(), "lr": 0.003}])               # ← новое: у критика шаг больше
    played, total, history = 0, 0.0, []
    s, _ = env.reset(seed=seed * 10_000)
    while played < steps:
        states, actions, rewards, s, terminated, truncated = play(actor, env, s, rng, n)  # ← новое: кусок из n шагов
        tail = 0.0 if terminated else critic(torch.as_tensor(s)).item()                   # ← новое: хвост — оценка критика
        G = torch.as_tensor(returns_to_go(rewards, tail=tail), dtype=torch.float32)       # ← новое: G_t с хвостом
        V = critic(torch.as_tensor(states)).squeeze(1)
        weights = (G - V).detach()
        loss_actor = -(weights * log_probs_of(actor, states, actions)).sum()
        loss_critic = ((G - V) ** 2).sum()
        loss = loss_actor + loss_critic
        opt.zero_grad(); loss.backward(); opt.step()
        played += len(rewards); total += rewards.sum()
        if terminated or truncated:                                                       # эпизод закончился — начинаем новый
            history.append((played, total)); total = 0.0
            s, _ = env.reset(seed=seed * 10_000 + len(history))
    return np.array(history), actor, critic
