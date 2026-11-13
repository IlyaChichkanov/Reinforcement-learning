"""Код обучения Q-сети из лекции 6 — без изменений, чтобы не повторять его в ноутбуке лекции 7.

Среда — CartPole: состояние s — четыре числа (x, ẋ, φ, φ̇), действия 0 (←) и 1 (→), награда +1 за шаг.
Переход — пятёрка (s, a, r, s′, конец); «конец» равен 1, если шест упал.

    make_qnet(seed=0)                 Q-сеть 4 → 64 → 64 → 2: состояние -> оценки двух действий
    epsilon_greedy(qnet, s, eps, rng) случайное действие с вероятностью ε, иначе действие с наибольшим Q
    to_tensors(transitions)           список переходов -> пять тензоров: S (n × 4), A (n), R (n), S2 (n × 4), D (n)
    td_targets(net, R, S2, D, gamma=0.99)   цель y = r + γ (1 − конец) max Q_net(s′) — тензор из n чисел
    q_of(qnet, S, A)                  Q(s_t, a_t) для каждого перехода — тензор из n чисел
    q_max(net, s)                     max Q(s, a) для одного состояния — число
    real_return(length, gamma=0.99)   сумма наград эпизода из length шагов с дисконтом: 1 + γ + … + γ^(length − 1)
    train_qnet(seed, replay=True, episodes=300, lr=1e-3, batch=64)
        Q-сеть лекции 6: цель считает сама обучаемая сеть. replay=True — шаг на пачке из буфера опыта,
        replay=False — на одном последнем переходе. Возвращает (длины эпизодов, оценки старта, сеть).

Обвязка опытов лекции 7
    train_runs(train, seeds, **settings)
        запускает train(seed, **settings) для каждого seed; возвращает длины эпизодов и оценки старта —
        массивы «запуски × эпизоды» — и список обученных сетей
    print_summary(lengths, q_start, skip=100)
        таблица по вариантам: средняя длина эпизода после первых skip эпизодов и наибольшая оценка старта
    best_run(lengths, last=50)        номер запуска с самыми длинными последними эпизодами
"""
from collections import deque

import numpy as np
import torch

from dqn_plots import make_cartpole


def make_qnet(seed=0):
    # Q-сеть: состояние (4 числа) -> оценки двух действий (Q(s, ←), Q(s, →)); её веса — параметры θ
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 2))


def epsilon_greedy(qnet, s, eps, rng):
    # с вероятностью ε — случайное действие, иначе действие с наибольшим Q
    if rng.random() < eps:
        return int(rng.integers(2))
    with torch.no_grad():
        return int(qnet(torch.as_tensor(s)).argmax())


def to_tensors(transitions):
    # список переходов (s, a, r, s′, конец) -> пять тензоров: S (n × 4), A (n), R (n), S2 (n × 4), D (n)
    return [torch.as_tensor(np.array(column)) for column in zip(*transitions)]


def td_targets(net, R, S2, D, gamma=0.99):
    # цель y для каждого перехода: тензор из n чисел; net(S2) — таблица n × 2, max по действиям; без градиента
    with torch.no_grad():
        return R + gamma * (1 - D) * net(S2).max(dim=1).values


def q_of(qnet, S, A):
    # Q_θ(s_t, a_t) для каждого перехода: тензор из n чисел — из строки t таблицы qnet(S) столбец A[t]
    return qnet(S)[torch.arange(len(A)), A]


def q_max(net, s):
    # max_a Q(s, a) для одного состояния — просто число
    with torch.no_grad():
        return net(torch.as_tensor(s)).max().item()


def real_return(length, gamma=0.99):
    # сколько эпизод принёс на самом деле: 1 + γ + … + γ^(length − 1)
    return (1 - gamma ** np.asarray(length, dtype=float)) / (1 - gamma)


def train_qnet(seed, replay=True, episodes=300, lr=1e-3, batch=64):
    rng, env, qnet = np.random.default_rng(seed), make_cartpole(), make_qnet(seed)
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = deque(maxlen=50_000)                                     # журнал переходов
    step, lengths, q_start = 0, [], []
    for episode in range(episodes):
        s, _ = env.reset(seed=seed * 10_000 + episode)
        q_start.append(q_max(qnet, s))                                # во что сеть верит на старте эпизода
        for t in range(1, 501):
            # 1. шаг в среде, переход — в журнал; ε падает от 1 до 0.01 за первые 5000 шагов
            a = epsilon_greedy(qnet, s, eps=max(0.01, 1 - step / 5000), rng=rng)
            s2, r, terminated, truncated, _ = env.step(a)
            buffer.append((s, a, np.float32(r), s2, np.float32(terminated)))
            # 2. на чём учимся: 64 случайных перехода из журнала — или один последний
            picked = [buffer[i] for i in rng.integers(len(buffer), size=batch)] if replay else [buffer[-1]]
            S, A, R, S2, D = to_tensors(picked)
            # 3. цель считает сама обучаемая сеть; шаг градиента
            y = td_targets(qnet, R, S2, D)
            loss = ((y - q_of(qnet, S, A)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1; s = s2
            if terminated or truncated:
                break
        lengths.append(t)
    return np.array(lengths), np.array(q_start), qnet


def train_runs(train, seeds, **settings):
    results = [train(seed, **settings) for seed in seeds]
    lengths = np.array([result[0] for result in results])            # запуски × эпизоды
    q_start = np.array([result[1] for result in results])            # запуски × эпизоды
    return lengths, q_start, [result[-1] for result in results]


def print_summary(lengths, q_start, skip=100):
    numbers = lambda values: ", ".join(f"{v:.0f}" for v in values)
    head = f"шест после {skip} эпизодов, шагов"
    width = max(len(name) for name in lengths)
    print(f"{'вариант':{width}s}   {head}   наибольшая оценка старта      (число — один запуск)")
    for name in lengths:
        print(f"{name:{width}s}   {numbers(lengths[name][:, skip:].mean(axis=1)):{len(head)}s}   "
              f"{numbers(np.abs(q_start[name]).max(axis=1))}")


def best_run(lengths, last=50):
    return int(lengths[:, -last:].mean(axis=1).argmax())
