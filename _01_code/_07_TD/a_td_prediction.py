"""
시간차 예측 (Time Difference Prediction) - TD(0)

[강의 핵심: 시간차 학습 소개]
  - TD 학습은 타임 스텝마다 가치 함수 업데이트 (vs MC: 에피소드 종료 후)
  - 환경과 상호작용하면서 실시간으로 가치 함수 업데이트 가능
  - 부트스트랩(Bootstrap) 사용: 다른 상태의 가치 정보를 활용하여 추정

[MC vs TD 갱신 식 비교 - 강의 슬라이드 5~6페이지]
  - MC:  V(St) ← V(St) + α[ Gt          − V(St) ]   (타겟: 실제 누적보상 Gt)
  - TD:  V(St) ← V(St) + α[ R_{t+1} + γV(S_{t+1}) − V(St) ]   (타겟: TD 타겟)

[TD(0) 알고리즘 - 강의 슬라이드 7페이지 의사코드]
  초기화: V(s) ← 0,  V(terminal) = 0
  while True:
    s ← 초기 상태
    for each time step of episode:
      a ← π(s)
      r', s' ← 환경으로부터 관측
      if s' is terminal:
        V(s) ← V(s) + α[ r' − V(s) ]
      else:
        V(s) ← V(s) + α[ r' + γV(s') − V(s) ]   ← TD 타겟
      s ← s'

[TD와 MC/DP 비교 - 강의 슬라이드 8~11페이지]
  DP  : Model-free X (full-width),  Bootstrap O, Time-step 업데이트
  MC  : Model-free O (Sample),      Bootstrap X, End-of-episode 업데이트
  TD  : Model-free O (Sample),      Bootstrap O, Time-step 업데이트  ← 가장 범용적
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA       = 0.99      # 감가율
N_EPISODES  = 200_000    # 에피소드 수
ALPHA       = 0.01      # 학습률 (스텝 사이즈)

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH  = 4, 4
N_STATES       = HEIGHT * WIDTH   # 16
N_ACTIONS      = 4                # LEFT=0, DOWN=1, RIGHT=2, UP=3
ACTION_SYMBOLS = ['←', '↓', '→', '↑']

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s 의 타일 종류 반환"""
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]

def policy_action(s):
    """균등 랜덤 정책: 4개 방향(LEFT, DOWN, RIGHT, UP) 중 무작위 선택"""
    return np.random.randint(N_ACTIONS)

# ── TD(0) 예측 ─────────────────────────────────────────────────
def td_prediction(env, n_episodes=N_EPISODES, alpha=ALPHA, gamma=GAMMA):
    """
    상태 가치 V(s) 예측하는 테이블 기반 TD(0) 학습
    (강의 슬라이드 7페이지 의사코드 구현)

    TD 타겟:    R_{t+1} + γ · V(S_{t+1})
    TD 오차:    δt = R_{t+1} + γV(S_{t+1}) − V(St)
    갱신 식:    V(St) ← V(St) + α · δt

    Args:
        env:        환경
        n_episodes: 총 에피소드 수
        alpha:      학습률
        gamma:      감가율

    Returns:
        V                 (np.ndarray): 수렴된 V(s)     [N_STATES]
        td_errors_history (list):       에피소드별 평균 TD 오차
    """
    # 초기화: V(s) ← 0,  V(terminal) = 0 (슬라이드 의사코드 1~3행)
    V = np.zeros(N_STATES)

    td_errors_history = []

    for ep in range(n_episodes):
        # ── 슬라이드 의사코드 5행: Initialize s ───────────────
        state, _ = env.reset()
        ep_td_errors = []

        # ── 슬라이드 의사코드 6~15행: for each time step ──────
        while True:
            # 7행: a ← π(s)
            action = policy_action(state)

            # 8행: Take action a, observe r', s'
            next_state, reward, terminated, truncated, _ = env.step(action)

            # ── TD 타겟 계산 및 V(s) 갱신 ─────────────────────
            # 강의: V(St) ← V(St) + α[R_{t+1} + γV(S_{t+1}) − V(St)]
            if terminated:
                # 9~10행: if s' is terminal → V(s) ← V(s) + α[r' − V(s)]
                td_target = reward
            else:
                # 11~12행: else → V(s) ← V(s) + α[r' + γV(s') − V(s)]
                td_target = reward + gamma * V[next_state]

            td_error  = td_target - V[state]   # δt (TD 오차)
            V[state] += alpha * td_error        # 갱신

            ep_td_errors.append(abs(td_error))

            # 14행: s ← s'
            state = next_state
            if terminated or truncated:
                break

        td_errors_history.append(np.mean(ep_td_errors))

        # ── 진행 상황 출력 ────────────────────────────────────
        if (ep + 1) % 10_000 == 0:
            print(f"  Episode {ep+1:>6} | α={alpha:.4f} | γ={gamma:.2f} | "
                  f"TD Error (Avg): {td_errors_history[-1]:.6f}")

    print(f"\n[TD(0) 예측 완료]")
    print(f"  총 에피소드 수: {n_episodes}")
    print(f"  학습률 α:       {alpha}")

    return V, td_errors_history


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(V):
    print("\n" + "=" * 55)
    print("  수렴된 상태 가치 함수 V(s)")
    print("=" * 55)
    print(f"  {'State':<12}  {'Tile':>6}  {'V(s)':>10}")
    print("  " + "-" * 33)
    for s in range(N_STATES):
        r, c  = divmod(s, WIDTH)
        tile  = get_tile(s)
        label = f"({r},{c})"
        print(f"  {label:<12}  {tile:>6}  {V[s]:>10.6f}")

    print("\n  상태 가치 격자 V(s):")
    print("  ┌─────────────────────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            s    = r * WIDTH + c
            tile = get_tile(s)
            if tile in ('H', 'G'):
                row_str += f" {tile:^7} "
            else:
                row_str += f" {V[s]:6.3f}  "
        print(row_str + "│")
    print("  └─────────────────────────────────┘")


# ── 시각화 ────────────────────────────────────────────────────
def visualize(V, td_errors_history):
    fig, (ax_v, ax_err) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "TD(0) Prediction  (Temporal-Difference Learning)\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  α={ALPHA}  |  "
        f"Episodes={N_EPISODES}",
        fontsize=13, fontweight='bold'
    )

    # ── (A) 상태 가치 함수 열지도 ───────────────────────────────
    v_grid = np.zeros((HEIGHT, WIDTH))
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        tile = get_tile(s)
        if tile not in ('H', 'G'):
            v_grid[r, c] = V[s]

    vmax = max(V.max(), 0.01)
    im = ax_v.imshow(v_grid, cmap='YlOrRd', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax_v, fraction=0.046, pad=0.04)

    colors_cell = {'H': '#37474F', 'G': '#1565C0'}
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        tile = get_tile(s)
        if tile in ('H', 'G'):
            ax_v.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1,
                           facecolor=colors_cell[tile], zorder=2,
                           edgecolor='white', linewidth=2))
            ax_v.text(c, r, tile, ha='center', va='center',
                      fontsize=14, color='white', fontweight='bold', zorder=3)
        else:
            norm_val = V[s] / vmax if vmax > 0 else 0
            txt_color = 'white' if norm_val > 0.55 else 'black'
            ax_v.text(c, r, f"{V[s]:.3f}", ha='center', va='center',
                      fontsize=10, color=txt_color, fontweight='bold', zorder=3)

    ax_v.set_xlim(-0.5, WIDTH - 0.5)
    ax_v.set_ylim(HEIGHT - 0.5, -0.5)
    ax_v.set_xticks(range(WIDTH));  ax_v.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_v.set_yticks(range(HEIGHT)); ax_v.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_v.set_title("(A) State Value Function  V(s)\n"
                   "TD Target: R_{t+1} + γV(S_{t+1})",
                   fontsize=10, fontweight='bold')
    ax_v.set_aspect('equal')

    # ── (B) TD 오차 감소 곡선 ───────────────────────────────────
    window = 500
    if len(td_errors_history) >= window:
        err_avg = np.convolve(td_errors_history,
                               np.ones(window) / window, mode='valid')
        ax_err.plot(err_avg, color='#E53935', linewidth=1.5,
                    label=f'TD Error Moving Avg (window={window})')
    ax_err.set_xlabel("Episode", fontsize=10)
    ax_err.set_ylabel("|δt|  (TD Error)", fontsize=10)
    ax_err.set_title("(B) TD Error  |δt| = |R_{t+1} + γV(S_{t+1}) − V(St)|\n"
                     "TD Error Decreases as Convergence Progresses",
                     fontsize=10, fontweight='bold')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./a_td_prediction_img.png', dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] a_td_prediction_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  시간차 예측 (Time Difference Prediction) - TD(0)")
    print("  [갱신 식: V(St) ← V(St) + α[R_{t+1} + γV(S_{t+1}) − V(St)]]")
    print("  [Model-free  |  Bootstrap O  |  Time-step 업데이트]")
    print("=" * 60)

    env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

    V, td_errors_history = td_prediction(env)

    print_results(V)
    visualize(V, td_errors_history)

    env.close()


if __name__ == '__main__':
    main()
