# Reinforcement Learning

#### Что это за курс

Курс даёт теоретическую базу и практический опыт в обучении с подкреплением (RL) — от табличных методов и MDP до глубокого RL, offline RL, multi-agent RL и RLHF для LLM. Цель — чтобы студент мог не только реализовать и отладить типовые RL-алгоритмы с нуля, но и осознанно выбирать метод под задачу, понимать его слабые места и читать современные статьи по теме.

#### Люди

* Лектор: Илья Чичканов

#### Формат

* 15 недель, 1 занятие в неделю (пара из лекции + семинара, суммарно ~1.5 "пары" контактного времени)
* Лекция — теория и разбор алгоритмов
* Семинар — практика: разбор кода, эксперименты в Jupyter, живое программирование
* Домашнее задание — почти каждую неделю, в формате `.ipynb` с автопроверкой части заданий через `assert`

#### Пререквизиты

* **Математика**: линейная алгебра, теория вероятностей и математическая статистика, основы оптимизации (градиентный спуск), начала матанализа
* **Программирование**: Python (numpy, базовый PyTorch), умение отлаживать код
* **ML/DL**: базовый курс машинного обучения; знакомство с нейросетями и backpropagation приветствуется, но не обязательно — необходимый минимум по DL даём по ходу курса

#### Инструменты

* Python 3.10+, [Gymnasium](https://gymnasium.farama.org/) для сред, [PyTorch](https://pytorch.org/) для нейросетевых методов
* Jupyter notebooks для лекций/семинаров/домашек
* Библиотеки для сравнения и вдохновения: Stable-Baselines3, CleanRL (код читаем, но домашки просим реализовывать самостоятельно)

#### Оценивание (черновой вариант, обсуждаем на первой лекции)

* 60% — домашние задания (устроены неравномерно по сложности, вес домашки указывается в самой домашке)
* 30% — итоговый проект (неделя 15: защита)
* 10% — активность на семинарах

#### Итоговый проект

Мини-исследование или реализация RL-алгоритма/агента на среде по выбору (Gymnasium classic control / MuJoCo / Atari / кастомная среда / RL для LLM-дообучения). Требования и список тем — см. `Projects.md` (появится к неделе 8, когда будет пройден нужный минимум методов).

---

# Программа

## Неделя 1 (01.09): Введение в RL, многорукие бандиты, формализация MDP

### Лекция

* Как устроен курс: формат, оценивание, инструменты
* Зачем нужен RL: примеры с демо и ссылками (TD-Gammon, DQN, AlphaGo/AlphaZero, OpenAI Five, робототехника, дата-центры, RLHF и рассуждающие LLM)
* Что такое RL и чем он отличается от supervised/unsupervised learning
* Живое демо в Gymnasium: CartPole со случайной и эвристической политикой
* Agent-environment interaction loop, эпизодические и непрерывные задачи
* Многорукие бандиты: постановка задачи, exploration vs exploitation
* ε-greedy, UCB1, Thompson Sampling, regret и его анализ (интуиция, без строгих доказательств)
* Марковский процесс принятия решений (MDP): состояния, действия, награды, переходы, дисконтирование
* Policy, return, value function V(s), action-value function Q(s,a)
* Уравнения Беллмана (ожидания) — вывод и интуиция

### Семинар

* Знакомство с Gymnasium: интерфейс `reset/step`, пространства состояний и действий
* Реализация среды многорукого бандита с нуля
* Реализация ε-greedy, UCB1, Thompson Sampling; сравнение regret на графиках
* Первое знакомство с табличной MDP-средой (FrozenLake / кастомный GridWorld)
* Мини-семинар по PyTorch (`seminar/pytorch_intro.ipynb`): тензоры, autograd, `nn.Module`, цикл обучения, behavior cloning эвристики на CartPole

### Домашнее задание

* Реализовать и сравнить бандит-алгоритмы (ε-greedy с расписанием, UCB1, Thompson Sampling для Bernoulli-бандита) на нескольких конфигурациях рук
* Теоретическая часть: вывод уравнения Беллмана для V и Q, вычисление return для простых MDP вручную

## Неделя 2 (08.09): Динамическое программирование

### Лекция

* Bellman optimality equations
* Policy Evaluation (iterative), Policy Improvement, Policy Iteration
* Value Iteration, сходимость (интуиция через contraction mapping)
* Generalized Policy Iteration
* Ограничения DP: требуется полная модель среды, curse of dimensionality

### Семинар

* Реализация Policy Iteration и Value Iteration для FrozenLake и кастомного GridWorld
* Визуализация V(s) и оптимальной политики по шагам сходимости

### Домашнее задание

* Реализовать Policy Iteration и Value Iteration с нуля, сравнить скорость сходимости
* Применить к задаче "Frozen Lake" большего размера и к задаче про управление запасами (inventory management) как MDP с непрерывными состояниями, дискретизированными вручную

## Неделя 3 (15.09): Monte Carlo и Temporal Difference обучение

### Лекция

* Model-free prediction: Monte Carlo policy evaluation (every-visit, first-visit)
* TD(0), TD-error, сравнение MC vs TD (bias/variance, online обучение)
* Model-free control: on-policy (SARSA) vs off-policy (Q-learning)
* Expected SARSA, максимизационное смещение (maximization bias) и Double Q-learning

### Семинар

* Реализация SARSA и Q-learning на FrozenLake / Cliff Walking
* Сравнение траекторий обучения on-policy vs off-policy (классический пример Cliff Walking)

### Домашнее задание

* Реализовать SARSA, Q-learning, Double Q-learning на Cliff Walking и Taxi-v3
* Проанализировать влияние ε-расписания и learning rate на сходимость

## Неделя 4 (22.09): Аппроксимация функций, TD(λ)

### Лекция

* Зачем нужна аппроксимация: большие/непрерывные пространства состояний
* Линейная аппроксимация ценности, feature engineering (tile coding, RBF)
* Semi-gradient TD(0), semi-gradient SARSA
* Eligibility traces, TD(λ), forward view vs backward view
* On-policy vs off-policy с аппроксимацией: проблема "deadly triad" (function approximation + bootstrapping + off-policy)

### Семинар

* Линейная аппроксимация Q-функции с tile coding на MountainCar
* Реализация SARSA(λ) с eligibility traces

### Домашнее задание

* Реализовать semi-gradient SARSA и SARSA(λ) с tile coding на MountainCar-v0
* Экспериментально показать deadly triad на упрощённом примере (Baird's counterexample)

## Неделя 5 (29.09): Deep Q-Learning

### Лекция

* От линейной аппроксимации к нейросетям: DQN (Mnih et al., 2015)
* Experience Replay, target network — зачем и как решают проблему нестабильности
* Double DQN, Dueling DQN architecture
* Prioritized Experience Replay
* Rainbow: объединение улучшений, ablation-анализ вклада каждого компонента

### Семинар

* Реализация DQN на CartPole с нуля (replay buffer, target network)
* Добавление Double DQN и Dueling-архитектуры, сравнение кривых обучения

### Домашнее задание

* Реализовать DQN + Double DQN + Dueling DQN на LunarLander-v2
* (бонус) добавить Prioritized Experience Replay и сравнить sample efficiency

## Неделя 6 (06.10): Policy Gradient методы

### Лекция

* Почему value-based методов недостаточно: стохастические/непрерывные политики
* Policy Gradient Theorem, вывод REINFORCE
* Baseline для снижения дисперсии (value function baseline)
* Actor-Critic: разделение ролей актора и критика
* A2C (advantage actor-critic), синхронный vs асинхронный (A3C) сбор опыта

### Семинар

* Реализация REINFORCE с baseline на CartPole
* Реализация Advantage Actor-Critic (A2C) с несколькими параллельными средами

### Домашнее задание

* Реализовать REINFORCE (с baseline и без) и A2C на CartPole/LunarLander
* Сравнить дисперсию градиента и стабильность обучения между методами

## Неделя 7 (13.10): TRPO и PPO

### Лекция

* Проблема выбора шага в policy gradient методах, monotonic improvement
* Trust Region Policy Optimization (TRPO): KL-ограничение, natural gradient (интуиция)
* Generalized Advantage Estimation (GAE): компромисс bias/variance
* Proximal Policy Optimization (PPO): clipped surrogate objective, почему он вытеснил TRPO на практике
* Практические детали PPO: normalize advantages, value clipping, entropy bonus

### Семинар

* Реализация PPO с нуля (clipped objective + GAE) на CartPole/LunarLander
* Разбор "37 implementation details of PPO" — какие детали реально важны

### Домашнее задание

* Реализовать PPO с нуля и обучить на LunarLander-v2 и BipedalWalker-v3 (или аналогичной среде)
* Ablation: убрать/добавить 2-3 implementation detail и показать эффект на кривой обучения

## Неделя 8 (20.10): Непрерывное управление — DDPG, TD3, SAC

### Лекция

* Специфика непрерывных action spaces, детерминированная политика
* DDPG: deterministic policy gradient, actor-critic с replay buffer
* Проблемы DDPG: переоценка Q, хрупкость к гиперпараметрам
* TD3: clipped double-Q, delayed policy updates, target policy smoothing
* SAC: maximum entropy RL, автоматическая настройка температуры, почему SAC устойчивее на практике

### Семинар

* Реализация DDPG/TD3 на Pendulum-v1
* Реализация SAC, сравнение с TD3 на непрерывной среде

### Домашнее задание

* Реализовать TD3 и SAC на среде типа HalfCheetah/Hopper (MuJoCo или Brax) или Pendulum+BipedalWalker при отсутствии MuJoCo
* Сравнить sample efficiency и итоговый результат

## Неделя 9 (27.10): Model-Based RL

### Лекция

* Model-free vs model-based: компромисс sample efficiency vs asymptotic performance
* Обучение модели среды (динамики), проблема накопления ошибки при роллаутах
* Dyna-Q: совмещение real experience и planning
* Model Predictive Control (MPC) с обученной моделью
* MBPO и современные гибридные подходы; краткое введение в world models (Dreamer) как мост к неделе 13-14

### Семинар

* Реализация Dyna-Q на GridWorld, сравнение с Q-learning по числу реальных взаимодействий
* Обучение простой модели динамики (нейросеть) и MPC-планирование на CartPole/Pendulum

### Домашнее задание

* Реализовать Dyna-Q и показать выигрыш в sample efficiency относительно Q-learning
* Реализовать простой MBPO-подобный pipeline (обученная модель + короткие роллауты + SAC/PPO) на Pendulum

## Неделя 10 (03.11): Exploration в глубоком RL

### Лекция

* Почему ε-greedy недостаточно в средах со sparse reward
* Count-based exploration и псевдо-подсчёты (pseudo-counts) в больших пространствах
* Intrinsic motivation: curiosity (ICM), Random Network Distillation (RND)
* Bootstrapped DQN, uncertainty-based exploration
* UCB/Thompson Sampling идеи, перенесённые в deep RL (bandit-подобные бонусы)

### Семинар

* Реализация RND-бонуса поверх PPO на среде со sparse reward (например, MountainCar или кастомная sparse-reward среда)
* Визуализация покрытия пространства состояний с exploration bonus и без

### Домашнее задание

* Добавить curiosity/RND-бонус к своему PPO/DQN агенту из недель 5-7 и показать эффект на sparse-reward версии среды
* Сравнить с epsilon-greedy baseline

## Неделя 11 (10.11): Imitation Learning и Offline RL

### Лекция

* Behavioral Cloning: постановка, проблема covariate shift
* DAgger: интерактивный сбор данных с экспертом
* GAIL: imitation learning через adversarial training (связь с GAN)
* Offline RL: обучение по фиксированному датасету без взаимодействия со средой
* Проблема distributional shift и переоценки Q вне поддержки датасета
* BCQ, CQL, IQL — как современные методы борются с extrapolation error

### Семинар

* Сбор экспертных траекторий и обучение BC на CartPole/LunarLander
* Реализация DAgger, сравнение с чистым BC
* Разбор кода CQL/IQL на готовом offline-датасете (D4RL-подобном)

### Домашнее задание

* Реализовать BC и DAgger, сравнить качество и число обращений к эксперту
* Реализовать CQL или IQL на offline-датасете и сравнить с naive BC/behavior cloning на том же датасете

## Неделя 12 (17.11): Multi-Agent RL

### Лекция

* От single-agent к multi-agent: кооперация, конкуренция, смешанные игры
* Independent learners и проблема нестационарности среды с точки зрения одного агента
* Centralized Training, Decentralized Execution (CTDE)
* QMIX (value decomposition), MAPPO (multi-agent PPO)
* Self-play и его роль в достижении сверхчеловеческой игры (AlphaGo/AlphaStar, кратко)

### Семинар

* Реализация independent Q-learning/PPO на простой multi-agent среде (например, PettingZoo)
* Реализация QMIX или MAPPO на кооперативной задаче

### Домашнее задание

* Реализовать MAPPO (или QMIX) на кооперативной multi-agent среде из PettingZoo
* Сравнить с independent learners baseline по итоговому качеству координации

## Неделя 13 (24.11): Distributional RL, иерархический и Meta-RL, POMDP

### Лекция

* Distributional RL: зачем моделировать распределение возврата, а не только среднее
* C51, Quantile Regression DQN (QR-DQN), IQN — кратко
* Hierarchical RL: options framework, feudal networks — идея временной абстракции
* Meta-RL: обучение агента, который быстро адаптируется к новой задаче (RL², MAML-подход к RL, кратко)
* Частично наблюдаемые MDP (POMDP): recurrent policies, belief state (интуиция)

### Семинар

* Реализация C51 или QR-DQN поверх DQN-инфраструктуры недели 5, сравнение на Atari-подобной задаче
* Recurrent PPO (LSTM-политика) на среде с частичной наблюдаемостью

### Домашнее задание

* Реализовать QR-DQN и сравнить с обычным DQN по стабильности и итоговому качеству
* (на выбор) добавить рекуррентность в PPO и обучить на POMDP-версии CartPole (с замаскированной скоростью)

## Неделя 14 (01.12): RL для LLM — RLHF и альтернативы

### Лекция

* От классического RL к дообучению языковых моделей: постановка задачи, action space = токены
* Reward modeling: сбор предпочтений (pairwise comparisons), обучение reward model, Bradley-Terry модель
* RLHF pipeline: SFT → reward model → PPO с KL-штрафом к SFT-политике
* Проблемы RLHF на практике: reward hacking, нестабильность PPO для LLM, дороговизна
* DPO (Direct Preference Optimization) и его вывод как "RL без RL"
* Альтернативы и развитие: RLAIF, IPO/KTO (кратко), RL для reasoning (GRPO/PPO с verifiable rewards)

### Семинар

* Реализация упрощённого RLHF-пайплайна на маленькой модели: reward model + PPO с KL-штрафом на игрушечной задаче (например, sentiment-контролируемая генерация)
* Реализация DPO на том же датасете предпочтений, сравнение с PPO-версией

### Домашнее задание

* Реализовать DPO дообучение маленькой языковой модели на датасете предпочтений
* Сравнить (качественно и по reward model) результат с PPO-RLHF версией из семинара

## Неделя 15 (08.12): Фронтир RL, защита проектов

### Лекция

* Обзор открытых проблем: sample efficiency, sim-to-real, safe RL, credit assignment на длинных горизонтах
* RL в реальных приложениях: рекомендательные системы, робототехника, LLM-агенты, RL для оптимизации инфраструктуры
* Куда идти дальше: ключевые лаборатории, конференции (NeurIPS/ICML/ICLR RL-треки), как читать современные RL-статьи

### Семинар / Практика

* Защита итоговых проектов: короткие доклады студентов + обсуждение

### Домашнее задание

* Нет — вместо этого сдача итогового проекта

---

# Литература

## Основные

* Richard S. Sutton, Andrew G. Barto, [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) ❗️ — основной учебник для недель 1-4
* David Silver, [UCL Course on RL](https://www.davidsilver.uk/teaching/) (видео + слайды) ❗️
* Sergey Levine, [CS285: Deep Reinforcement Learning (Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/) ❗️ — для недель 5-13
* [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) ❗️ — отличные конспекты + чистый код по policy gradient/PPO/SAC/TD3

## По отдельным темам

* Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (DQN)
* Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
* Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3)
* Janner et al., [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) (MBPO)
* Kumar et al., [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779)
* Rashid et al., [QMIX](https://arxiv.org/abs/1803.11485); Yu et al., [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) (MAPPO)
* Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT/RLHF)
* Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
* Huang et al., [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

# TODO

* Уточнить оценивание и дедлайны совместно со студентами на первой лекции
* Определить, будет ли доступ к GPU/кластеру для домашек 5-14 и добавить инструкцию в `tools/`
* Написать `Projects.md` к неделе 8
