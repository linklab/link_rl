"""
ε-탐욕적 정책 기반 On-policy 몬테카를로 제어
(On-policy Monte Carlo Control with ε-Greedy Policy)

[핵심 원리]
  - 행동 정책(behavior policy)과 목표 정책(target policy)이 동일 (On-policy)
  - ε-탐욕 정책으로 에피소드 생성 → 같은 ε-탐욕 정책을 향상
  - 매 에피소드 후 Q(s,a) 갱신 → argmax_a Q(s,a) 방향으로 ε-탐욕 정책 갱신

[알고리즘 개요]
  초기화: Q(s,a) ← 0,  Returns(s,a) ← [],  π ← ε-탐욕(Q)
  for 에피소드 in range(N_EPISODES):
    1. 현재 ε-탐욕 정책 π 로 에피소드 생성: (S0,A0,R1), ..., (ST-1,AT-1,RT)
    2. G ← 0
    3. for t = T-1, T-2, ..., 0:  (역방향 순회)
         G ← γ·G + R_{t+1}
         First-visit: (St, At) 가 처음 등장한 경우에만
           Returns(St,At).append(G)
           Q(St,At) ← mean(Returns(St,At))
           π(St) ← argmax_a Q(St,a) 기준 ε-탐욕 정책

[First-visit vs Every-visit]
  - 본 코드는 First-visit MC 사용
  - 에피소드 내 첫 번째 방문(St, At)만 G를 반영

[ε-탐욕 정책]
  π(a|s) = 1 - ε + ε/|A|   if a == argmax_a Q(s,a)
           ε/|A|             otherwise
  → ε 감소 스케줄: ε_t = max(ε_min, ε_start / (1 + decay * episode))
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA       = 0.99     # 감가율
N_EPISODES  = 200_000   # 에피소드 수
EPSILON_START = 0.9    # 초기 ε
EPSILON_MIN   = 0.05   # 최소 ε
EPSILON_DECAY = 0.0001 # ε 감소 속도

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH   = 4, 4
N_STATES        = HEIGHT * WIDTH   # 16
N_ACTIONS       = 4               # LEFT=0, DOWN=1, RIGHT=2, UP=3
ACTION_SYMBOLS  = ['←', '↓', '→', '↑']

# ── 환경 생성 ──────────────────────────────────────────────────
env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s 의 타일 종류 반환"""
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]

def epsilon_greedy_action(Q, s, epsilon):
    """
    ε-탐욕 정책: 확률 ε 로 무작위 행동, 확률 1-ε 로 Q 최대 행동 선택

    Args:
        Q (np.ndarray): 상태-행동 가치 함수  (shape: N_STATES × N_ACTIONS)
        s (int):        현재 상태
        epsilon (float): 탐색 확률

    Returns:
        action (int)
    """
    if np.random.rand() < epsilon:
        return env.action_space.sample()      # 탐색 (무작위)
    else:
        return int(np.argmax(Q[s]))           # 활용 (탐욕)

def generate_episode(Q, epsilon):
    """
    현재 ε-탐욕 정책으로 에피소드 1개 생성

    Returns:
        episode (list of (state, action, reward))
    """
    episode = []
    state, _ = env.reset()
    while True:
        action = epsilon_greedy_action(Q, state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode


# ── On-policy MC 제어 (First-visit) ────────────────────────────
def on_policy_mc_control(env, n_episodes=N_EPISODES,
                          gamma=GAMMA,
                          epsilon_start=EPSILON_START,
                          epsilon_min=EPSILON_MIN,
                          epsilon_decay=EPSILON_DECAY):
    """
    ε-탐욕 정책 기반 On-policy MC 제어 (First-visit)

    Args:
        env:            Gymnasium 환경
        n_episodes:     총 에피소드 수
        gamma:          감가율
        epsilon_start:  초기 ε
        epsilon_min:    최소 ε
        epsilon_decay:  ε 감소 계수

    Returns:
        Q               (np.ndarray): 수렴된 Q(s,a)  [N_STATES × N_ACTIONS]
        policy          (np.ndarray): 최적 탐욕 정책  [N_STATES]
        returns_sum     (np.ndarray): Returns 합계    [N_STATES × N_ACTIONS]
        returns_cnt     (np.ndarray): Returns 방문 수 [N_STATES × N_ACTIONS]
        episode_rewards (list):       에피소드별 누적 보상
        epsilon_history (list):       에피소드별 ε 값
    """
    Q            = np.zeros((N_STATES, N_ACTIONS))
    returns_sum  = np.zeros((N_STATES, N_ACTIONS))
    returns_cnt  = np.zeros((N_STATES, N_ACTIONS), dtype=int)

    episode_rewards = []
    epsilon_history = []

    for ep in range(n_episodes):
        # ε 감소 스케줄
        epsilon = max(epsilon_min, epsilon_start / (1.0 + epsilon_decay * ep))
        epsilon_history.append(epsilon)

        # ── 1. 에피소드 생성 ─────────────────────────────────
        episode = generate_episode(Q, epsilon)
        ep_reward = sum(r for _, _, r in episode)
        episode_rewards.append(ep_reward)

        # ── 2. 역방향 순회로 G 계산 & First-visit Q 갱신 ─────
        G = 0.0
        visited = set()   # First-visit 판별용

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:          # First-visit
                visited.add((s, a))
                returns_sum[s, a] += G
                returns_cnt[s, a] += 1
                Q[s, a] = returns_sum[s, a] / returns_cnt[s, a]

        # ── 3. 진행 상황 출력 ─────────────────────────────────
        if (ep + 1) % 10_000 == 0:
            avg_r = np.mean(episode_rewards[-10_000:])
            print(f"  Episode {ep+1:>6} | ε={epsilon:.4f} | "
                  f"Avg Reward (last 10k): {avg_r:.4f}")

    # ── 최적 탐욕 정책 산출 (ε=0) ──────────────────────────────
    policy = np.array([int(np.argmax(Q[s])) for s in range(N_STATES)])

    print(f"\n[On-policy MC 제어 완료]")
    print(f"  총 에피소드 수: {n_episodes}")
    print(f"  최종 ε:        {epsilon_history[-1]:.4f}")

    return Q, policy, returns_sum, returns_cnt, episode_rewards, epsilon_history


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(Q, policy):
    print("\n" + "=" * 60)
    print("  수렴된 Q(s,a) 및 최적 정책 π*(s)")
    print("=" * 60)
    print(f"  {'State':<12}  {'←(L)':>8}  {'↓(D)':>8}  {'→(R)':>8}  {'↑(U)':>8}  {'π*':>4}")
    print("  " + "-" * 56)
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
def visualize(Q, policy, episode_rewards, epsilon_history):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        "On-policy MC Control  (ε-Greedy, First-visit)\n"
        f"FrozenLake 4×4  |  is_slippery=True  |  γ={GAMMA}  |  "
        f"Episodes={N_EPISODES}",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.50, wspace=0.40,
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
                     "argmax_a Q(s,a)  [Greedy]",
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
                    f"On-policy MC  |  ε-greedy",
                    fontsize=10, fontweight='bold')
    ax_rw.legend(fontsize=9)
    ax_rw.grid(True, alpha=0.3)

    # ── (C) ε 감소 곡선 ─────────────────────────────────────────
    ax_eps = fig.add_subplot(gs[0, 2])
    ax_eps.plot(epsilon_history, color='#E53935', linewidth=1.5,
                label='ε schedule')
    ax_eps.axhline(EPSILON_MIN, color='gray', linestyle='--', linewidth=1.2,
                   label=f'ε_min = {EPSILON_MIN}')
    ax_eps.set_xlabel("Episode", fontsize=10)
    ax_eps.set_ylabel("ε (epsilon)", fontsize=10)
    ax_eps.set_title(f"(C) ε Decay Schedule\n"
                     f"ε = max(ε_min, ε_start / (1 + decay × ep))",
                     fontsize=10, fontweight='bold')
    ax_eps.legend(fontsize=9)
    ax_eps.grid(True, alpha=0.3)

    plt.savefig('./a_on_policy_mc_control_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] a_on_policy_mc_control_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  ε-탐욕적 정책 기반 On-policy 몬테카를로 제어")
    print("  [행동 정책 = 목표 정책 = ε-탐욕(Q)]")
    print("  [First-visit MC  |  is_slippery=True]")
    print("=" * 60)

    (Q, policy,
     returns_sum, returns_cnt,
     episode_rewards, epsilon_history) = on_policy_mc_control(env)

    print_results(Q, policy)
    visualize(Q, policy, episode_rewards, epsilon_history)

    env.close()
