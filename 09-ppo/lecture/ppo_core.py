"""Служебный код лекции 9: лунный модуль и стратегия-сеть для PPO.

Среда и сети — те же, что в лекции 8 (`ac_core.py`), поэтому многое просто импортируется: в репозитории этот файл
лежит рядом с ноутбуком лекции 8, в Colab его скачивает первая ячейка.
Новое здесь — сбор куска, который запоминает log π старой стратегии: без этих чисел отношение ρ не посчитать.

    make_lander(continuous=False)         среда LunarLander-v3; continuous=True — действие это два числа от −1 до 1
    make_net(n_out, seed)                 сеть 8 → 64 → 64 → n_out (из лекции 8)
    collect(actor, env, s, rng, n)        кусок до n шагов или до конца эпизода: состояния, действия, награды,
                                          log π старой стратегии, следующее состояние, terminated, truncated
    returns_to_go(rewards, gamma, tail)   G_t с хвостом (из лекции 8)
    log_probs_of(actor, states, actions)  log π(a_t | s_t) текущей стратегии (из лекции 8)

Непрерывный вариант (раздел 5):

    log_prob_normal(mu, log_sigma, a)     логарифм плотности нормального распределения в точке a (сумма по числам действия)
    collect_continuous(actor, log_sigma, env, s, rng, n)    то же, что collect, но действие берётся из нормального распределения

Сохранённые результаты (lander_ppo_results.npz) — чтобы ноутбук не обучал модуль заново:

    load_runs(kind)                       истории обучения по запускам: kind — "plain1" (один проход),
                                          "plain10" (10 проходов без клиппинга), "ppo" или "cont" (непрерывный модуль);
                                          история — массив «эпизоды × 2»: (шагов в среде сделано, сумма наград эпизода)
    load_nets()                           стратегия-сеть и критик, обученные на прошлой лекции (из файла результатов)
    measure_rho(epochs, clip)             опыт: один кусок (эпизод), epochs проходов по нему -> ρ каждого шага
    print_runs(name, histories)           сводка варианта по запускам
    steps_to_land(history, level, window) сколько шагов в среде сделано, когда среднее window эпизодов впервые достигло level
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "08-actor-critic" / "lecture"))
from ac_core import log_probs_of, make_net, returns_to_go, steps_to_land  # noqa: E402,F401
import gymnasium as gym  # noqa: E402

RESULTS = Path(__file__).with_name("lander_ppo_results.npz")


def make_lander(continuous=False, render=False):
    return gym.make("LunarLander-v3", continuous=continuous, render_mode="rgb_array" if render else None)


def collect(actor, env, s, rng, n):
    """Кусок до n шагов текущей стратегией; раньше — если эпизод кончился.
    Кроме состояний, действий и наград запоминает log π(a_t | s_t) той стратегии, которая кусок сыграла."""
    states, actions, rewards, logp_old = [], [], [], []
    for _ in range(n):
        with torch.no_grad():
            logp_all = torch.log_softmax(actor(torch.as_tensor(s)), dim=-1)
            probs = logp_all.exp().numpy().astype(float)
        a = int(rng.choice(len(probs), p=probs / probs.sum()))
        s2, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r); logp_old.append(float(logp_all[a]))
        s = s2
        if terminated or truncated:
            break
    return (np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(logp_old, dtype=np.float32), s, terminated, truncated)


# ------------------------------------------------------------------ непрерывные действия (раздел 5)

def log_prob_normal(mu, log_sigma, a):
    """Логарифм плотности нормального распределения со средним mu и разбросом exp(log_sigma) в точке a."""
    sigma = log_sigma.exp()
    return (-0.5 * ((a - mu) / sigma) ** 2 - log_sigma - 0.5 * np.log(2 * np.pi)).sum(dim=-1)


def collect_continuous(actor, log_sigma, env, s, rng, n):
    """То же, что collect, но действие — два числа из нормального распределения со средним actor(s)."""
    states, actions, rewards, logp_old = [], [], [], []
    terminated = truncated = False
    for _ in range(n):
        with torch.no_grad():
            mu = actor(torch.as_tensor(s))
            a = mu + log_sigma.exp() * torch.as_tensor(rng.standard_normal(len(mu)), dtype=torch.float32)
            logp = log_prob_normal(mu, log_sigma, a)
        s2, r, terminated, truncated, _ = env.step(np.clip(a.numpy(), -1, 1).astype(np.float32))
        states.append(s); actions.append(a.numpy()); rewards.append(r); logp_old.append(float(logp))
        s = s2
        if terminated or truncated:
            break
    return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32),
            np.array(logp_old, dtype=np.float32), s, terminated, truncated)


def play_frames(actor, seed, log_sigma=None):
    """Один эпизод для анимации: кадры и сумма наград. log_sigma задан — непрерывная среда."""
    env = make_lander(continuous=log_sigma is not None, render=True)
    rng = np.random.default_rng(seed)
    s, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        with torch.no_grad():
            out = actor(torch.as_tensor(s))
            a = (np.clip(out.numpy(), -1, 1).astype(np.float32) if log_sigma is not None else int(out.argmax()))
        s, r, terminated, truncated, _ = env.step(a)
        frames.append(env.render()); total += r
        done = terminated or truncated
    return frames, total, truncated


# ------------------------------------------------------------------ сохранённые результаты

def load_runs(kind):
    data = np.load(RESULTS)
    return [data[name].astype(float) for name in sorted(k for k in data.files if k.startswith(f"{kind}_history_"))]


def print_runs(name, histories):
    """Сводка варианта по запускам: среднее последних 100 эпизодов и сколько шагов ушло до среднего 200."""
    last = np.array([history[-100:, 1].mean() for history in histories])
    landed = sorted(round(steps / 1000) for steps in map(steps_to_land, histories) if steps is not None)
    print(name)
    print(f"    среднее последних 100 эпизодов: {last.mean():.0f}; по запускам: {' '.join(f'{x:.0f}' for x in np.sort(last))};"
          f" не меньше 200 — в {(last >= 200).sum()} из {len(last)}")
    print(f"    до среднего 200 дошли {len(landed)} из {len(last)} запусков"
          + (f", шагов в среде, тыс.: {' '.join(map(str, landed))}" if landed else ""))


def load_nets():
    """Стратегия-сеть и критик, обученные актором-критиком на прошлой лекции (веса лежат в файле результатов)."""
    data = np.load(RESULTS)
    actor, critic = make_net(4, 0), make_net(1, 0)
    actor.load_state_dict({k: torch.as_tensor(data[f"actor_{k}"]) for k in
                           (name[len("actor_"):] for name in data.files if name.startswith("actor_"))})
    critic.load_state_dict({k: torch.as_tensor(data[f"critic_{k}"]) for k in
                            (name[len("critic_"):] for name in data.files if name.startswith("critic_"))})
    return actor, critic


def measure_rho(epochs=10, clip=False, eps=0.2, seed=0, n=2048, lr=3e-4, lr_c=1e-3, gamma=0.99):
    """Опыт раздела 1: взять обученную стратегию, сыграть ею один кусок (эпизод), сделать по нему epochs проходов
    и посмотреть, во сколько раз изменилась вероятность каждого сыгранного действия: ρ = π_новая(a|s) / π_старая(a|s).
    Шаг обучения тот же, что в train: это ровно первое обновление train, только сети уже обучены."""
    actor, critic = load_nets()                                        # каждый вызов начинает с одних и тех же весов
    rng, env = np.random.default_rng(seed), make_lander()
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": lr},
                            {"params": critic.parameters(), "lr": lr_c}])
    s, _ = env.reset(seed=seed * 10_000)
    states, actions, rewards, logp_old, s2, terminated, truncated = collect(actor, env, s, rng, n)
    tail = 0.0 if terminated else critic(torch.as_tensor(s2)).item()
    G = torch.as_tensor(returns_to_go(rewards, gamma, tail=tail), dtype=torch.float32)
    states, actions, logp_old = torch.as_tensor(states), torch.as_tensor(actions), torch.as_tensor(logp_old)
    advantage = (G - critic(states).squeeze(1)).detach()
    for _ in range(epochs):
        ratio = torch.exp(log_probs_of(actor, states, actions) - logp_old)
        terms = ratio * advantage
        if clip:
            terms = torch.min(terms, ratio.clamp(1 - eps, 1 + eps) * advantage)
        loss = -terms.mean() + ((G - critic(states).squeeze(1)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.exp(log_probs_of(actor, states, actions) - logp_old).numpy()
