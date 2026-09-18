"""Служебный код лекции 2: модель озера как MDP и функции из лекции 1 (сыграть эпизод, return, Монте-Карло).

Ничего нового по сравнению с лекцией здесь нет — это те же функции, вынесенные из ноутбука,
чтобы в лекции остался только содержательный код.
"""
from collections import deque

import numpy as np

GAMMA = 0.95


def run_session(env, policy, rng, max_steps=100):
    """Один эпизод политикой-таблицей policy[s, a]. Возвращает состояния (включая последнее), действия и награды."""
    s, _ = env.reset(seed=int(rng.integers(1_000_000)))
    states, actions, rewards = [s], [], []
    for _ in range(max_steps):
        a = int(rng.choice(len(policy[s]), p=policy[s]))
        s, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        if terminated or truncated:
            break
    return states, actions, rewards


def discounted_return(rewards, gamma=GAMMA):
    """G = r_0 + γ r_1 + γ² r_2 + ..."""
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def estimate_values(sessions, n_states, gamma=GAMMA):
    """V(s) ≈ средний return эпизода, считая от первого попадания в клетку s (Монте-Карло, как в лекции 1)."""
    total, count = np.zeros(n_states), np.zeros(n_states)
    for states, actions, rewards in sessions:
        for t, s in enumerate(states[:-1]):
            if s not in states[:t]:
                total[s] += discounted_return(rewards[t:], gamma); count[s] += 1
    return np.divide(total, count, out=np.full(n_states, np.nan), where=count > 0)


def success_rate(env, policy, rng, n_episodes=1000, max_steps=200):
    """Доля эпизодов, в которых политика дошла до цели."""
    return float(np.mean([sum(run_session(env, policy, rng, max_steps)[2]) > 0 for _ in range(n_episodes)]))


def mdp_matrices(env):
    """Модель среды из таблицы внутри Frozen Lake: P[s, a, s'] — вероятности переходов, R[s, a] — ожидаемая награда за шаг."""
    u = env.unwrapped
    n_s, n_a = u.observation_space.n, u.action_space.n
    P, R = np.zeros((n_s, n_a, n_s)), np.zeros((n_s, n_a))
    for s in range(n_s):
        for a in range(n_a):
            for prob, s_next, reward, terminated in u.P[s][a]:
                P[s, a, s_next] += prob
                R[s, a] += prob * reward
    return P, R


def solve_q_star(P, R, gamma=GAMMA, tol=1e-12):
    """Оптимальная Q-функция: метод простой итерации для уравнения оптимальности Беллмана (то, что выводится в лекции)."""
    Q = np.zeros(P.shape[:2])
    for _ in range(100_000):
        Q_new = R + gamma * P @ Q.max(axis=1)
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new
    return Q


def greedy_policy(Q):
    """Детерминированная политика-таблица: в каждом состоянии действие с наибольшим Q."""
    return np.eye(Q.shape[1])[Q.argmax(axis=1)]


def shortest_path_policy(desc):
    """Детерминированная политика «кратчайший путь к G по нескользкому льду» (поиск в ширину от цели)."""
    nrow, ncol = len(desc), len(desc[0])
    moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    goal = next((r, c) for r in range(nrow) for c in range(ncol) if desc[r][c] == "G")
    best, queue = {goal: None}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for a, (dr, dc) in moves.items():                       # сосед, из которого действие a ведёт в (r, c)
            rr, cc = r - dr, c - dc
            if 0 <= rr < nrow and 0 <= cc < ncol and desc[rr][cc] not in "H" and (rr, cc) not in best:
                best[(rr, cc)] = a; queue.append((rr, cc))
    policy = np.ones((nrow * ncol, 4)) / 4
    for (r, c), a in best.items():
        if a is not None:
            policy[r * ncol + c] = np.eye(4)[a]
    return policy
