"""Клетчатый мир лекции 2: среда, её модель и функции для эпизодов.

Мир — классический пример из учебника ШАД (он же GridWorld из курса Berkeley AI):
сетка 3 x 4, слева внизу старт, справа вверху выход +1, под ним яма -1, в середине стена.
Ходить можно в четыре стороны, но ветер с вероятностью 0.1 сносит в каждую из боковых сторон;
за каждый шаг агент платит 0.04 — иначе выгодно гулять вечно.

Читать этот файл не обязательно: всё, что нужно на лекции, показано картинками.
"""
import numpy as np

GAMMA = 0.95
STEP_REWARD = -0.04
NOISE = 0.1
ARROWS = "←↓→↑"                       # действия: 0 = влево, 1 = вниз, 2 = вправо, 3 = вверх
ACTION_NAMES = ["влево", "вниз", "вправо", "вверх"]
LAYOUT = [
    "...G",                           # G — выход, награда +1
    ".#.X",                           # # — стена, X — яма, награда -1
    "S...",                           # S — старт
]
MOVES = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
SIDEWAYS = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}   # куда сносит ветер при каждом действии
TERMINAL_REWARD = {"G": 1.0, "X": -1.0}


class GridWorld:
    """Клетчатый мир с ветром. Интерфейс как у сред Gymnasium: reset() и step(action)."""

    def __init__(self, layout=LAYOUT, step_reward=STEP_REWARD, noise=NOISE):
        self.layout = [list(row) for row in layout]
        self.n_rows, self.n_cols = len(self.layout), len(self.layout[0])
        self.n_states, self.n_actions = self.n_rows * self.n_cols, 4
        self.step_reward, self.noise = step_reward, noise
        self.start = next(r * self.n_cols + c for r in range(self.n_rows) for c in range(self.n_cols)
                          if self.layout[r][c] == "S")
        self.rng = np.random.default_rng(0)

    # ---- служебное -------------------------------------------------------
    def char(self, s):
        return self.layout[s // self.n_cols][s % self.n_cols]

    def is_wall(self, s):
        return self.char(s) == "#"

    def is_terminal(self, s):
        return self.char(s) in TERMINAL_REWARD

    def _move(self, s, a):
        """Куда агент попадёт из клетки s, шагнув в направлении a (в стену или за край — останется на месте)."""
        r, c = divmod(s, self.n_cols)
        dr, dc = MOVES[a]
        rr, cc = r + dr, c + dc
        if 0 <= rr < self.n_rows and 0 <= cc < self.n_cols and self.layout[rr][cc] != "#":
            return rr * self.n_cols + cc
        return s

    def transitions(self, s, a):
        """Все исходы действия a в клетке s: список (вероятность, следующая клетка, награда, конец эпизода)."""
        if self.is_terminal(s) or self.is_wall(s):
            return [(1.0, s, 0.0, True)]
        out = {}
        for direction, prob in [(a, 1 - 2 * self.noise)] + [(d, self.noise) for d in SIDEWAYS[a]]:
            s_next = self._move(s, direction)
            out[s_next] = out.get(s_next, 0.0) + prob
        return [(prob, s_next, self.step_reward + TERMINAL_REWARD.get(self.char(s_next), 0.0), self.is_terminal(s_next))
                for s_next, prob in out.items()]

    # ---- интерфейс среды -------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = self.start
        return self.s, {}

    def step(self, action):
        outcomes = self.transitions(self.s, action)
        probs = [p for p, *_ in outcomes]
        prob, s_next, reward, terminated = outcomes[self.rng.choice(len(outcomes), p=probs)]
        self.s = s_next
        return s_next, reward, terminated, False, {}


def mdp_matrices(env):
    """Модель среды двумя массивами: P[s, a, s'] — вероятности переходов, R[s, a] — средняя награда за шаг."""
    P = np.zeros((env.n_states, env.n_actions, env.n_states))
    R = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            for prob, s_next, reward, _ in env.transitions(s, a):
                P[s, a, s_next] += prob
                R[s, a] += prob * reward
    return P, R


def run_episode(env, policy, rng, max_steps=100):
    """Один эпизод стратегией-таблицей policy[s, a]. Возвращает клетки (включая последнюю), действия и награды."""
    s, _ = env.reset(seed=int(rng.integers(1_000_000)))
    states, actions, rewards = [s], [], []
    for _ in range(max_steps):
        a = int(rng.choice(env.n_actions, p=policy[s]))
        s, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        if terminated or truncated:
            break
    return states, actions, rewards


def discounted_return(rewards, gamma=GAMMA):
    """G = r_0 + γ r_1 + γ² r_2 + ..."""
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def average_return(env, policy, rng, n_episodes=1000, gamma=GAMMA):
    """Средний return стратегии по n_episodes сыгранным эпизодам (оценка Монте-Карло из лекции 1)."""
    return float(np.mean([discounted_return(run_episode(env, policy, rng)[2], gamma) for _ in range(n_episodes)]))


def estimate_values_mc(sessions, n_states, gamma=GAMMA):
    """V(s) ≈ средний return, считая от первого попадания в клетку s."""
    total, count = np.zeros(n_states), np.zeros(n_states)
    for states, actions, rewards in sessions:
        for t, s in enumerate(states[:-1]):
            if s not in states[:t]:
                total[s] += discounted_return(rewards[t:], gamma); count[s] += 1
    return np.divide(total, count, out=np.full(n_states, np.nan), where=count > 0)


def uniform_policy(env):
    """Стратегия «бросаю кубик»: все четыре действия равновероятны."""
    return np.ones((env.n_states, env.n_actions)) / env.n_actions


def greedy_policy(Q):
    """Детерминированная стратегия-таблица: в каждой клетке действие с наибольшим Q."""
    return np.eye(Q.shape[1])[Q.argmax(axis=1)]


def solve_q_star(P, R, gamma=GAMMA, tol=1e-12):
    """Оптимальная Q-функция. Как именно она считается — раздел 4 лекции; здесь просто «ответ»."""
    Q = np.zeros(P.shape[:2])
    for _ in range(100_000):
        Q_new = R + gamma * P @ Q.max(axis=1)
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new
    return Q
