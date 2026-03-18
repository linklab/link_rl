"""
Weighted Importance Sampling 기반 Off-policy 몬테카를로 제어
(Off-policy Monte Carlo Control with Weighted Importance Sampling)

[핵심 원리]
  - 행동 정책(behavior policy) b ≠ 목표 정책(target policy) π  → Off-policy
  - b : 균등 무작위 정책  (탐색 보장, b(a|s) = 1/|A| = 1/4)
  - π : 탐욕 정책         (Q 최대 행동, π(a*|s) = 1)
  - 중요도 비율(Importance Sampling Ratio):
        ρ_{t:T-1} = Π_{k=t}^{T-1} [π(A_k|S_k) / b(A_k|S_k)]

[Weighted IS 추정 공식  ← PDF p.38]
  q_π(s,a) = Σ_{t∈J(s,a)} ρ_{t+1:T-1} · G_t  /  Σ_{t∈J(s,a)} ρ_{t+1:T-1}
             ──────────────────────────────────────────────────────────────
             분모: Σ W_i = 누적 가중치 합  (가중 평균)

[Weighted IS 점진적 갱신 유도  ← PDF p.40-42]
  C_n = Σ_{i=1}^{n} W_i           (누적 가중치 합)
  Q_n = Σ_{i=1}^{n} W_i·G_i / C_n (가중 평균)

  C_{n+1} = C_n + W_{n+1}
  Q_{n+1} = Q_n + (W_{n+1} / C_{n+1}) · (G_{n+1} − Q_n)

  ★ 여기서 분모 C_{n+1} = Σ W_i → C[s,a] += W (가중치 누적)

[Ordinary IS와의 핵심 차이  ← PDF p.35, 40]
  ┌──────────────────────────────────────────────────────────────┐
  │          │  Ordinary IS            │  Weighted IS            │
  ├──────────┼─────────────────────────┼─────────────────────────┤
  │ 추정공식 │ Σ(ρ·G) / n             │ Σ(ρ·G) / Σ(ρ)          │
  │ C[s,a]  │ 방문 횟수 n  (+=1)     │ 누적 가중치 Σ W  (+=W)  │
  │ 편향     │ 불편 (unbiased)         │ 편향 (biased)           │
  │ 분산     │ 높음 (ρ 폭발 위험)     │ 낮음 (안정적)           │
  │ 실용성   │ 이론적 우위             │ 실용적 우위 (권장)      │
  └──────────┴─────────────────────────┴─────────────────────────┘

[알고리즘 개요  ← Sutton & Barto 5.7절 / PDF p.44 의사코드]
  초기화: Q(s,a) ← 0,  C(s,a) ← 0,  π ← greedy(Q)
  for 에피소드:
    1. 행동 정책 b (균등 랜덤)로 에피소드 생성
    2. G ← 0,  W ← 1
    3. for t = T-1, T-2, ..., 0 (역방향):
         G       ← γ·G + R_{t+1}
         C[St,At] += W           ← ★ Weighted IS: 가중치 누적
         Q[St,At] += (W / C[St,At]) · (G − Q[St,At])
         π(St)   ← argmax_a Q(St,a)
         if At ≠ π(St): break
         W       ← W · [π(At|St) / b(At|St)] = W · N_ACTIONS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA      = 0.99       # 감가율
N_EPISODES = 200_000    # 에피소드 수

VALIDATION_EPISODES_INTERVAL = 100  # 검증 수행 간격 (훈련 에피소드 수)
VALIDATION_NUM_EPISODES      = 3       # 검증 시 수행 에피소드 수

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH  = 4, 4
N_STATES       = HEIGHT * WIDTH   # 16
N_ACTIONS      = 4                # LEFT=0, DOWN=1, RIGHT=2, UP=3
ACTION_SYMBOLS = ['←', '↓', '→', '↑']

# ── 환경 생성 ──────────────────────────────────────────────────
env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ── 행동 정책: 균등 무작위 ─────────────────────────────────────
BEHAVIOR_PROB = 1.0 / N_ACTIONS    # b(a|s) = 1/4

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s의 타일 종류 반환 (S/F/H/G)"""
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]


def generate_episode_behavior(env):
    """
    행동 정책 b (균등 무작위)로 에피소드 생성

    Returns:
        episode (list of (state, action, reward))
    """
    episode = []
    state, _ = env.reset()
    while True:
        action = env.action_space.sample()    # b: 균등 무작위
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode


def greedy_action(Q, s):
    """목표 정책 π: Q 최대 행동 (탐욕)"""
    return int(np.argmax(Q[s]))


def validate_policy(Q, n_episodes=VALIDATION_NUM_EPISODES):
    """
    현재 Q를 탐욕(ε=0) 정책으로 n_episodes 번 실행하여 평균 보상 반환

    Args:
        Q (np.ndarray): 상태-행동 가치 함수  (shape: N_STATES × N_ACTIONS)
        n_episodes (int): 검증 에피소드 수

    Returns:
        avg_reward (float): 탐욕 정책의 평균 누적 보상
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        while True:
            action = greedy_action(Q, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
    return total_reward / n_episodes


# ── Weighted IS Off-policy MC 제어 ────────────────────────────
def off_policy_mc_weighted_is(env, n_episodes=N_EPISODES, gamma=GAMMA):
    """
    Weighted Importance Sampling 기반 Off-policy MC 제어
    (Sutton & Barto Algorithm 5.7절 기반)

    핵심: C[s,a]는 누적 가중치 합 Σ W_i  → C[s,a] += W
          (Ordinary IS와의 결정적 차이)

    증분 갱신 공식  ← PDF p.40 수식:
      C[s,a] += W                              ← 가중치 합 누적
      Q[s,a] += (W / C[s,a]) * (G - Q[s,a])   ← 가중 평균 갱신

    Returns:
        Q                          (ndarray [N_STATES × N_ACTIONS]): 수렴된 Q(s,a)
        policy                     (ndarray [N_STATES]):             최적 탐욕 정책
        C                          (ndarray [N_STATES × N_ACTIONS]): 누적 가중치 합 Σ W_i
        episode_rewards            (list): 에피소드별 누적 보상
        validation_episode_rewards (list): 검증 에피소드별 평균 보상
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    # ★ Weighted IS: C[s,a] = 누적 가중치 합 Σ W_i  (Ordinary IS: 방문 횟수 n)
    C = np.zeros((N_STATES, N_ACTIONS), dtype=float)

    episode_rewards            = []
    validation_episode_rewards = []

    for ep in range(n_episodes):
        # ── 1. 행동 정책 b로 에피소드 생성 ──────────────────
        episode   = generate_episode_behavior(env)
        ep_reward = sum(r for _, _, r in episode)
        episode_rewards.append(ep_reward)

        # ── 2. 역방향 순회 ────────────────────────────────
        G = 0.0
        W = 1.0    # 누적 중요도 비율 ρ_{t:T-1}, 초기값 = 1

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]

            # 2-1. 반환값(Return) 계산
            G = gamma * G + r

            # 2-2. ★ Weighted IS 핵심: C[s,a] += W  (가중치 누적)
            #      Ordinary IS라면 C[s,a] += 1 (방문 횟수 카운트)
            C[s, a] += W

            # 2-3. Q 증분 갱신
            #      Q(s,a) ← Q(s,a) + (W / Σ W) · (G − Q(s,a))
            #      = 가중 평균의 점진적 계산  (PDF p.40-42 수식)
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            # 2-4. 목표 정책 갱신: π(s) ← argmax_a Q(s,a)
            pi_a = greedy_action(Q, s)

            # 2-5. 목표 정책 이탈 여부 확인 → 수학적으로 필수인 early-exit
            #
            #  [왜 break가 필요한가?]
            #  중요도 비율 W는 역방향으로 누적 곱셈됩니다:
            #    W = Π_{k=t}^{T-1} [ π(A_k|S_k) / b(A_k|S_k) ]
            #
            #  목표 정책 π는 탐욕 정책(결정적)이므로:
            #    - A_k == π(S_k) 이면 → π(A_k|S_k) = 1  → 비율 = 1/b > 0
            #    - A_k != π(S_k) 이면 → π(A_k|S_k) = 0  → 비율 = 0  → W = 0
            #
            #  W = 0 이 되면, 이 에피소드의 모든 이전 타임스텝(더 앞 방향)에서
            #  Q 갱신 항 (W / C) * (G - Q) 역시 0이 되어 갱신이 의미 없어집니다.
            #
            #  따라서 A_t != π(S_t) 인 순간 역방향 순회를 중단(break)하는 것은
            #  수학적으로 완전히 동치이며, 불필요한 반복 연산도 줄여 줍니다.
            #
            #  ※ Q와 π는 break 이전에 이미 갱신된 상태이므로 정보 손실 없음.
            if a != pi_a:
                break

            # 2-6. 중요도 비율 누적
            #      π(a|s) = 1   (탐욕 정책: 최적 행동 확률 1)
            #      b(a|s) = 1/N_ACTIONS (균등 정책)
            #      W ← W · π(a|s) / b(a|s) = W · N_ACTIONS
            W *= 1.0 / BEHAVIOR_PROB    # = W * N_ACTIONS

        if (ep + 1) % VALIDATION_EPISODES_INTERVAL == 0:
            train_avg = np.mean(episode_rewards[-VALIDATION_EPISODES_INTERVAL:])
            val_avg   = validate_policy(Q, VALIDATION_NUM_EPISODES)
            validation_episode_rewards.append(val_avg)
            print(f"  Episode {ep+1:>6} | "
                  f"Train Episode Reward (Avg): {train_avg:.4f} | "
                  f"Validation Episode Reward (Avg): {val_avg:.4f}")

    # ── 3. 최적 탐욕 정책 산출 ──────────────────────────────
    policy = np.array([greedy_action(Q, s) for s in range(N_STATES)])

    print(f"\n[Weighted IS Off-policy MC 제어 완료]")
    print(f"  총 에피소드 수             : {n_episodes}")
    print(f"  C (누적 가중치 합 Σ W) 최대: {C.max():.2f}")

    return Q, policy, C, episode_rewards, validation_episode_rewards


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(Q, policy, C):
    print("\n" + "=" * 70)
    print("  수렴된 Q(s,a) 및 최적 정책 [Weighted IS Off-policy MC]")
    print("=" * 70)
    print(f"  {'State':<12}  {'←(L)':>8}  {'↓(D)':>8}  {'→(R)':>8}  {'↑(U)':>8}  {'π*':>4}")
    print("  " + "-" * 62)
    for s in range(N_STATES):
        r, c  = divmod(s, WIDTH)
        tile  = get_tile(s)
        label = f"({r},{c})[{tile}]"
        if tile in ('H', 'G'):
            print(f"  {label:<12}  {'terminal':>8}")
        else:
            best = int(np.argmax(Q[s]))
            row  = f"  {label:<12}"
            for a in range(N_ACTIONS):
                marker = "*" if a == best else " "
                row   += f"  {Q[s,a]:7.4f}{marker}"
            row += f"  {ACTION_SYMBOLS[best]:>4}"
            print(row)

    print("\n  최적 정책 격자:")
    print("  ┌──────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            s    = r * WIDTH + c
            tile = get_tile(s)
            if tile == 'H':   row_str += "  H "
            elif tile == 'G': row_str += "  G "
            else:             row_str += f"  {ACTION_SYMBOLS[policy[s]]} "
        print(row_str + " │")
    print("  └──────────────────┘")


# ── 시각화 ────────────────────────────────────────────────────
def visualize(Q, policy, C, episode_rewards, validation_episode_rewards):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        "Off-policy MC Control  ─  Weighted Importance Sampling\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  "
        f"Episodes={N_EPISODES:,}\n"
        "Behavior Policy b: Uniform Random  |  Target Policy π: Greedy(Q)\n"
        "Q ← Σ(W·G) / Σ(W)   [C += W, Biased · Low Variance, Recommended]",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.55, wspace=0.40,
                           top=0.72, bottom=0.12)

    # ── (A) 최적 정책 격자 ──────────────────────────────────
    ax_pol = fig.add_subplot(gs[0, 0])
    colors_cell = {'S': '#E8F5E9', 'F': '#E8F5E9',
                   'H': '#37474F', 'G': '#1565C0'}
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        tile = get_tile(s)
        ax_pol.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                         facecolor=colors_cell[tile], zorder=1,
                         edgecolor='white', linewidth=2))
        if tile == 'H':
            ax_pol.text(c, r, 'H\n(Hole)', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold', zorder=3)
        elif tile == 'G':
            ax_pol.text(c, r, 'G\n(Goal)', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold', zorder=3)
        else:
            ax_pol.text(c, r, ACTION_SYMBOLS[policy[s]], ha='center', va='center',
                        fontsize=18, color='#1976D2', fontweight='bold', zorder=3)
    ax_pol.set_xlim(-0.5, WIDTH - 0.5)
    ax_pol.set_ylim(HEIGHT - 0.5, -0.5)
    ax_pol.set_xticks(range(WIDTH))
    ax_pol.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_pol.set_yticks(range(HEIGHT))
    ax_pol.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_pol.set_title("(A) Optimal Policy  π*(s)\n"
                     "argmax_a Q(s,a)  [Greedy Target]",
                     fontsize=10, fontweight='bold')
    ax_pol.set_aspect('equal')

    # ── (B) 훈련 에피소드 보상 이동평균 ─────────────────────
    ax_rw = fig.add_subplot(gs[0, 1])
    window = 1000
    moving_avg = np.convolve(episode_rewards,
                             np.ones(window) / window, mode='valid')
    ax_rw.plot(moving_avg, color='#1976D2', linewidth=1.5,
               label=f'Train Moving Avg (window={window})')
    ax_rw.set_xlabel("Episode", fontsize=10)
    ax_rw.set_ylabel("Avg Reward", fontsize=10)
    ax_rw.set_title(f"(B) Train Episode Reward  (Moving Avg {window})\n"
                    "Weighted IS  |  Low Variance, Stable Convergence",
                    fontsize=10, fontweight='bold')
    ax_rw.legend(fontsize=9)
    ax_rw.grid(True, alpha=0.3)

    # ── (C) 검증 에피소드 평균 보상 ─────────────────────────
    ax_val = fig.add_subplot(gs[0, 2])
    val_x = [(i + 1) * VALIDATION_EPISODES_INTERVAL
             for i in range(len(validation_episode_rewards))]
    ax_val.plot(val_x, validation_episode_rewards, color='#43A047',
                linewidth=1.5, marker='o', markersize=4,
                label=f'Validation Avg ({VALIDATION_NUM_EPISODES} eps, greedy)')
    ax_val.set_xlabel("Episode", fontsize=10)
    ax_val.set_ylabel("Avg Reward", fontsize=10)
    ax_val.set_title(f"(C) Validation Reward (Greedy Policy, every {VALIDATION_EPISODES_INTERVAL//1000}k ep)\n"
                     f"ε=0  |  {VALIDATION_NUM_EPISODES} episodes per validation",
                     fontsize=10, fontweight='bold')
    ax_val.legend(fontsize=9)
    ax_val.grid(True, alpha=0.3)

    plt.savefig('./c_off_policy_mc_weighted_is_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] c_off_policy_mc_weighted_is_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Weighted IS  Off-policy MC 제어")
    print("  행동 정책 b : 균등 무작위   b(a|s) = 1/4")
    print("  목표 정책 π : 탐욕(Q)       π(a*|s) = 1")
    print("  추정 공식   : Q ← Σ(W·G) / Σ(W)  [C += W, 편향·저분산]")
    print("=" * 65)

    (Q, policy, C,
     episode_rewards,
     validation_episode_rewards) = off_policy_mc_weighted_is(env)
    print_results(Q, policy, C)
    visualize(Q, policy, C, episode_rewards, validation_episode_rewards)
    env.close()
