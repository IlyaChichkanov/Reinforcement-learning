"""Две функции обучения лекции 9 — тот же текст, что в ячейках ноутбука (файл собирается вместе с ноутбуком).

    train(seed, steps=STEPS, n=2048, epochs=10, eps=0.2, clip=True)
        дискретный модуль; epochs — сколько проходов по куску, clip — обрезать ли слагаемые шагов.
        Три варианта лекции: (epochs=1, clip=False), (epochs=10, clip=False), (epochs=10, clip=True) — это PPO.
    train_continuous(seed, steps=STEPS, n=2048, epochs=10, eps=0.2, sigma=0.5)
        тот же PPO на непрерывном модуле (действие — два числа от −1 до 1).

История — массив «эпизоды × 2»: (сколько шагов в среде сделано к концу эпизода, сумма наград эпизода).
Нужен этот файл для assets/make_animations.py: по нему посчитан lander_ppo_results.npz.
"""
import numpy as np
import torch

from ppo_core import (collect, collect_continuous, log_prob_normal, log_probs_of, make_lander, make_net,
                      returns_to_go)

STEPS = 1_000_000          # шагов в среде на один запуск


def train(seed, steps=STEPS, n=2048, epochs=10, eps=0.2, clip=True):
    # кусок играется один раз, потом по нему делается epochs проходов; clip=False — то же, но без обрезки
    rng, env = np.random.default_rng(seed), make_lander()
    actor, critic = make_net(4, seed), make_net(1, seed + 1000)
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": 0.0003},
                            {"params": critic.parameters(), "lr": 0.001}])
    played, total, history = 0, 0.0, []
    s, _ = env.reset(seed=seed * 10_000)
    while played < steps:
        # ← новое: collect запоминает ещё и log π сыгранных действий — logp_old
        states, actions, rewards, logp_old, s, terminated, truncated = collect(actor, env, s, rng, n)
        tail = 0.0 if terminated else critic(torch.as_tensor(s)).item()
        G = torch.as_tensor(returns_to_go(rewards, tail=tail), dtype=torch.float32)
        states, actions, logp_old = torch.as_tensor(states), torch.as_tensor(actions), torch.as_tensor(logp_old)
        advantage = (G - critic(states).squeeze(1)).detach()                    # вес действия A_t, как в лекции 8
        for _ in range(epochs):                                                 # ← новое: несколько проходов по куску
            ratio = torch.exp(log_probs_of(actor, states, actions) - logp_old)  # ← новое: ρ_t = π_новая / π_старая
            terms = ratio * advantage                                           # слагаемые шагов: ρ_t · A_t
            if clip:                                                            # ← новое: меньшее из двух чисел
                terms = torch.min(terms, ratio.clamp(1 - eps, 1 + eps) * advantage)
            loss_actor = -terms.mean()                                          # среднее: шаг не зависит от длины куска
            loss_critic = ((G - critic(states).squeeze(1)) ** 2).mean()
            loss = loss_actor + loss_critic
            opt.zero_grad(); loss.backward(); opt.step()
        played += len(rewards); total += rewards.sum()
        if terminated or truncated:
            history.append((played, total)); total = 0.0
            s, _ = env.reset(seed=seed * 10_000 + len(history))
    return np.array(history), actor, critic


def train_continuous(seed, steps=STEPS, n=2048, epochs=10, eps=0.2, sigma=0.5):
    # тот же PPO на непрерывном модуле: действие — два числа от −1 до 1
    rng, env = np.random.default_rng(seed), make_lander(continuous=True)              # ← новое: среда
    actor, critic = make_net(2, seed), make_net(1, seed + 1000)                       # ← новое: на выходе два средних
    log_sigma = torch.full((2,), float(np.log(sigma)), requires_grad=True)            # ← новое: разброс — тоже параметры
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": 0.0003},
                            {"params": [log_sigma], "lr": 0.0003},
                            {"params": critic.parameters(), "lr": 0.001}])
    played, total, history = 0, 0.0, []
    s, _ = env.reset(seed=seed * 10_000)
    while played < steps:
        # ← новое: действие берётся из нормального распределения
        states, actions, rewards, logp_old, s, terminated, truncated = collect_continuous(actor, log_sigma, env, s, rng, n)
        tail = 0.0 if terminated else critic(torch.as_tensor(s)).item()
        G = torch.as_tensor(returns_to_go(rewards, tail=tail), dtype=torch.float32)
        states, actions, logp_old = torch.as_tensor(states), torch.as_tensor(actions), torch.as_tensor(logp_old)
        advantage = (G - critic(states).squeeze(1)).detach()
        for _ in range(epochs):
            log_pi = log_prob_normal(actor(states), log_sigma, actions)               # ← новое: log π — логарифм плотности
            ratio = torch.exp(log_pi - logp_old)
            terms = torch.min(ratio * advantage, ratio.clamp(1 - eps, 1 + eps) * advantage)
            loss_actor = -terms.mean()
            loss_critic = ((G - critic(states).squeeze(1)) ** 2).mean()
            loss = loss_actor + loss_critic
            opt.zero_grad(); loss.backward(); opt.step()
        played += len(rewards); total += rewards.sum()
        if terminated or truncated:
            history.append((played, total)); total = 0.0
            s, _ = env.reset(seed=seed * 10_000 + len(history))
    return np.array(history), actor, critic, log_sigma.detach()
