"""
Ordinary Importance Sampling 기반 Off-policy 몬테카를로 제어
(Off-policy Monte Carlo Control with Ordinary Importance Sampling)

[핵심 원리]
  - 행동 정책(behavior policy) b ≠ 목표 정책(target policy) π  → Off-policy
  - b: 균등 무작위 정책  (탐색 보장)
  - π: 탐욕 정책  (Q 최대 행동)
  - 중요도 비율(Importance Sampling Ratio):
      ρ_{t:T-1} = Π_{k=t}^{T-1} [π(A_k|S_k) / b(A_k|S_k)]
  - Ordinary IS 추정:
      Q(s,a) ← Σ_i [ρ_i · G_i] / n   (n: 방문 횟수, 비가중 평균)

[알고리즘 개요]
  초기화: Q(s,a) ← 0,  C(s,a) ← 0 (누적 ρ·weight),  π ← 탐욕(Q)
  for 에피소드 in range(N_EPISODES):
    1. 행동 정책 b (균등 랜덤)로 에피소드 생성
    2. G ← 0,  W ← 1   (W: 누적 중요도 비율)
    3. for t = T-1, T-2, ..., 0:  (역방향)
         G  ← γ·G + R_{t+1}
         C[St,At] += W                        (누적 중요도)
         Q[St,At] += W/C[St,At] · (G - Q[St,At])  (증분 갱신)
         π(St) ← argmax_a Q(St,a)            (탐욕 정책 갱신)
         if At ≠ π(St): break                (목표 정책 벗어남 → 에피소드 종료)
         W  ← W · π(At|St) / b(At|St)
            = W · 1 / (1/|A|)  = W · |A|    (π 탐욕, b 균등)

[Ordinary IS의 특성]
  - 불편 추정량(unbiased estimator): E[ρ·G] = v_π(s)
  - 분산이 크다: ρ 가 매우 크거나 0에 가까울 수 있음
  - 가중 IS 에 비해 샘플 수 적을 때 불안정할 수 있음

[Weighted IS와의 차이]
  Ordinary IS: Q ← Σ(ρ_i · G_i) / n              (단순 평균)
  Weighted IS: Q ← Σ(ρ_i · G_i) / Σ(ρ_i)        (가중 평균)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA      = 0.99      # 감가율
N_EPISODES = 200_000    # 에피소드 수

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH   = 4, 4
N_STATES        = HEIGHT * WIDTH   # 16
N_ACTIONS       = 4               # LEFT=0, DOWN=1, RIGHT=2, UP=3
ACTION_SYMBOLS  = ['←', '↓', '→', '↑']

# ── 환경 생성 ──────────────────────────────────────────────────
env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ── 행동 정책: 균등 무작위 ─────────────────────────────────────
BEHAVIOR_PROB = 1.0 / N_ACTIONS    # b(a|s) = 1/4

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s 의 타일 종류 반환"""
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


# ── Ordinary IS Off-policy MC 제어 ────────────────────────────
def off_policy_mc_ordinary_is(env, n_episodes=N_EPISODES, gamma=GAMMA):
    """
    Ordinary Importance Sampling 기반 Off-policy MC 제어

    [증분 갱신 공식 유도]
      Ordinary IS:  Q_n(s,a) = Σ_{i=1}^{n} [W_i · G_i] / n
      증분:         Q_{n+1}  = Q_n + (W/C_n) · (G - Q_n)
                   (C_n = 방문 횟수 n, W = 누적 중요도 비율)

    Args:
        env:        Gymnasium 환경
        n_episodes: 총 에피소드 수
        gamma:      감가율

    Returns:
        Q               (np.ndarray): 수렴된 Q(s,a)  [N_STATES × N_ACTIONS]
        policy          (np.ndarray): 최적 탐욕 정책  [N_STATES]
        C               (np.ndarray): 누적 중요도 비율 합계
        episode_rewards (list):       에피소드별 누적 보상
        is_ratios       (list):       에피소드별 최종 W (중요도 비율 추적)
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    C = np.zeros((N_STATES, N_ACTIONS))   # 누적 중요도 비율 (Ordinary IS: 방문 횟수)

    episode_rewards = []
    is_ratios       = []

    for ep in range(n_episodes):
        # ── 1. 행동 정책 b 로 에피소드 생성 ──────────────────
        episode    = generate_episode_behavior(env)
        ep_reward  = sum(r for _, _, r in episode)
        episode_rewards.append(ep_reward)

        # ── 2. 역방향 순회: G 계산 & Ordinary IS Q 갱신 ──────
        G = 0.0
        W = 1.0   # 누적 중요도 비율: ρ_{t:T-1}

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            # ── Ordinary IS 증분 갱신 ──────────────────────
            #   C[s,a] : 방문 횟수 (n)
            #   Q[s,a] += W / C[s,a] * (G - Q[s,a])
            C[s, a] += W                           # W=1 이면 단순 카운트 증가
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            # 목표 정책 갱신: π(s) ← argmax_a Q(s,a)
            pi_a = greedy_action(Q, s)

            # 목표 정책과 다른 행동이면 이 이전 스텝은 더 이상 반영 불가 → break
            if a != pi_a:
                break

            # ── 중요도 비율 갱신 ────────────────────────────
            #   π(a|s) = 1 (탐욕 정책: 최적 행동 확률 = 1)
            #   b(a|s) = 1/N_ACTIONS (균등 정책)
            #   W ← W * [π(a|s) / b(a|s)] = W * N_ACTIONS
            W *= 1.0 / BEHAVIOR_PROB    # = W * N_ACTIONS

        is_ratios.append(W)

        if (ep + 1) % 10_000 == 0:
            avg_r = np.mean(episode_rewards[-10_000:])
            print(f"  Episode {ep+1:>6} | "
                  f"Avg Reward (last 10k): {avg_r:.4f} | "
                  f"W: {W:.2e}")

    # ── 최적 탐욕 정책 산출 ─────────────────────────────────────
    policy = np.array([greedy_action(Q, s) for s in range(N_STATES)])

    print(f"\n[Ordinary IS Off-policy MC 제어 완료]")
    print(f"  총 에피소드 수: {n_episodes}")
    print(f"  평균 중요도 비율 W: {np.mean(is_ratios):.4f}")

    return Q, policy, C, episode_rewards, is_ratios


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(Q, policy, C):
    print("\n" + "=" * 70)
    print("  수렴된 Q(s,a) 및 최적 정책 [Ordinary IS Off-policy MC]")
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
def visualize(Q, policy, C, episode_rewards, is_ratios):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        "Off-policy MC Control  (Ordinary Importance Sampling)\n"
        f"FrozenLake 4×4  |  is_slippery=True  |  γ={GAMMA}  |  "
        f"Episodes={N_EPISODES}\n"
        "Behavior Policy: Uniform Random  |  Target Policy: Greedy(Q)",
        fontsize=12, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.55, wspace=0.40,
                          top=0.76, bottom=0.12)

    # ── (A) 최적 정책 격자 ──────────────────────────────────────
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
                        fontsize=12, color='white', fontweight='bold', zorder=3)
        elif tile == 'G':
            ax_pol.text(c, r, 'G\n(Goal)', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold', zorder=3)
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

    # ── (B) 에피소드 보상 이동평균 ─────────────────────────────
    ax_rw = fig.add_subplot(gs[0, 1])
    window = 1000
    moving_avg = np.convolve(episode_rewards,
                              np.ones(window) / window, mode='valid')
    ax_rw.plot(moving_avg, color='#1976D2', linewidth=1.5,
               label=f'Moving Avg (window={window})')
    ax_rw.set_xlabel("Episode", fontsize=10)
    ax_rw.set_ylabel("Avg Reward", fontsize=10)
    ax_rw.set_title(f"(B) Episode Reward (Moving Avg {window})\n"
                    "Ordinary IS Off-policy MC",
                    fontsize=10, fontweight='bold')
    ax_rw.legend(fontsize=9)
    ax_rw.grid(True, alpha=0.3)

    # ── (C) 중요도 비율 (IS Ratio W) 이동평균 ───────────────────
    ax_is = fig.add_subplot(gs[0, 2])
    is_moving_avg = np.convolve(is_ratios,
                                np.ones(window) / window, mode='valid')
    ax_is.plot(is_moving_avg, color='#E53935', linewidth=1.5,
               label=f'Moving Avg (window={window})')
    ax_is.set_xlabel("Episode", fontsize=10)
    ax_is.set_ylabel("IS Ratio  W", fontsize=10)
    ax_is.set_title(f"(C) Importance Sampling Ratio  W\n"
                    "ρ = π(a|s) / b(a|s)  [Ordinary IS]",
                    fontsize=10, fontweight='bold')
    ax_is.legend(fontsize=9)
    ax_is.grid(True, alpha=0.3)

    plt.savefig('./b_off_policy_mc_ordinary_is_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] b_off_policy_mc_ordinary_is_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Ordinary Importance Sampling 기반 Off-policy MC 제어")
    print("  [행동 정책 b: 균등 무작위]  [목표 정책 π: 탐욕(Q)]")
    print("  [불편 추정 | 높은 분산 | Q ← Σ(ρ·G) / n]")
    print("=" * 65)

    (Q, policy,
     C, episode_rewards, is_ratios) = off_policy_mc_ordinary_is(env)

    print_results(Q, policy, C)
    visualize(Q, policy, C, episode_rewards, is_ratios)

    env.close()
