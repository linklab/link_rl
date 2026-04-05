"""
n-스텝 시간차 예측 (n-step Temporal-Difference Prediction)
FrozenLake 4×4 환경 기반 | n=1, n=2, n=4 비교

[강의 핵심: n-스텝 시간차 학습 - 슬라이드 13페이지]
  - 단일 스텝 TD 방법과 몬테카를로 방법의 중간에 위치
  - n이 작을수록 → 단일 스텝 TD(0)에 가까워짐
  - n이 커질수록 → 몬테카를로(MC) 방법에 가까워짐

  [개념적 위치]
  단일 스텝 TD(0) ←──── n=1 | n=2 | n=4 ────→ 몬테카를로 방법
  (One-step TD)                                 (Monte Carlo Method)

[n-스텝 누적 보상 (이득, Return) - 슬라이드 14~15페이지]
  1-스텝 이득:  G_{t:t+1} = R_{t+1} + γ · V(S_{t+1})
  2-스텝 이득:  G_{t:t+2} = R_{t+1} + γR_{t+2} + γ² · V(S_{t+2})
  n-스텝 이득:  G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γⁿ · V(S_{t+n})
                           (단, t+n ≥ T 이면 G_{t:t+n} = G_t ← 종료까지의 실제 보상)

[n-스텝 TD 갱신 식 - 슬라이드 15~16페이지]
  V(St) ← V(St) + α [ G_{t:t+n} − V(St) ]

  n=1 (TD(0)):  V(St) ← V(St) + α[ R_{t+1} + γV(S_{t+1}) − V(St) ]
  n=2:          V(St) ← V(St) + α[ R_{t+1} + γR_{t+2} + γ²V(S_{t+2}) − V(St) ]
  n=4:          V(St) ← V(St) + α[ R_{t+1} + ... + γ⁴V(S_{t+4}) − V(St) ]

[n-스텝 TD 예측 의사코드 - 슬라이드 17페이지]
  에피소드 종료 후 각 타임 스텝 t에 대해:
    G_{t:t+n} 계산 → V(St) 갱신
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA      = 0.99      # 감가율
N_EPISODES = 200_000   # 에피소드 수
ALPHA      = 0.01      # 학습률

# 비교할 n 값 목록 (슬라이드 16페이지: TD(0), n-step TD, MC 스펙트럼)
N_STEP_LIST = [1, 2, 4]

PRINT_INTERVAL = 10_000  # 콘솔 출력 간격

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

def generate_episode(env):
    """
    에피소드 1개 생성 → 전이 리스트 반환

    [슬라이드 17페이지: 에피소드 전체를 먼저 수집 후 갱신]

    Returns:
        transitions (list of (state, reward, next_state, terminated)):
            타임 스텝별 전이 정보
    """
    transitions = []
    state, _ = env.reset()
    while True:
        action = policy_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        transitions.append((state, reward, next_state, terminated))
        state = next_state
        if terminated or truncated:
            break
    return transitions


# ── n-스텝 이득 계산 (슬라이드 14~15페이지) ────────────────────
def compute_n_step_return(transitions, t, n, V, gamma):
    """
    시간 t에서 n-스텝 이득 G_{t:t+n} 계산

    [슬라이드 14페이지 공식]
    G_{t:t+n} = Σ_{k=0}^{min(n, T-t)-1} γ^k · R_{t+k+1}
              + γⁿ · V(S_{t+n})    (if t+n < T, 즉 에피소드 미종료)

    t+n ≥ T 이면:  G_{t:t+n} = 실제 누적 보상만 (부트스트랩 없음 → MC처럼)

    Args:
        transitions (list): (s, r, s', done) 튜플 리스트
        t    (int):         현재 타임 스텝 인덱스
        n    (int):         n-스텝 크기
        V    (np.ndarray):  현재 상태 가치 함수 [N_STATES]
        gamma (float):      감가율

    Returns:
        G (float): n-스텝 이득 G_{t:t+n}
    """
    T = len(transitions)
    G = 0.0

    # ── 실제 보상 누적: Σ_{k=0}^{min(n, T-t)-1} γ^k · R_{t+k+1} ──
    for k in range(min(n, T - t)):
        _, reward, _, _ = transitions[t + k]
        G += (gamma ** k) * reward

    # ── 부트스트랩: γⁿ · V(S_{t+n})  (t+n < T 인 경우만) ──────────
    # t+n >= T 이면 에피소드가 이미 종료 → 부트스트랩 없음 (슬라이드 14p)
    if t + n < T:
        _, _, next_state, done = transitions[t + n]
        if not done:
            G += (gamma ** n) * V[next_state]
        # done=True이면 terminal → V(terminal)=0 이므로 부트스트랩 생략

    return G


# ── n-스텝 TD 예측 (슬라이드 15~17페이지) ──────────────────────
def n_step_td_prediction(env, n, n_episodes=N_EPISODES,
                          alpha=ALPHA, gamma=GAMMA):
    """
    FrozenLake 환경에서 n-스텝 TD 예측 수행

    [슬라이드 17페이지 의사코드 구현]
    for 에피소드:
      에피소드 전체 수집
      for t = 0, 1, ..., T-1:
        G_{t:t+n} 계산 (슬라이드 14p 공식)
        V(St) ← V(St) + α[ G_{t:t+n} − V(St) ]  (슬라이드 15p 갱신 식)

    n=1: TD(0)   → R_{t+1} + γV(S_{t+1})
    n=2: 2-step  → R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
    n=4: 4-step  → R_{t+1} + ... + γ⁴V(S_{t+4})

    Args:
        env:        환경
        n    (int): n-스텝 크기
        n_episodes: 총 에피소드 수
        alpha:      학습률
        gamma:      감가율

    Returns:
        V                 (np.ndarray): 수렴된 V(s)     [N_STATES]
        td_errors_history (list):       에피소드별 평균 TD 오차 |G_{t:t+n} - V(St)|
    """
    # 초기화: V(s) ← 0,  V(terminal) = 0 (슬라이드 의사코드 1~3행)
    V = np.zeros(N_STATES)

    td_errors_history = []

    for ep in range(n_episodes):
        # ── 1. 에피소드 전체 생성 (슬라이드 17p: 에피소드 먼저 수집) ──
        transitions = generate_episode(env)
        T           = len(transitions)
        ep_td_errors = []

        # ── 2. 순방향 순회: 각 타임 스텝 t에서 V(St) 갱신 ──────────
        # [슬라이드 17페이지 의사코드]
        # V(St) ← V(St) + α[ G_{t:t+n} − V(St) ]
        for t in range(T):
            state, _, _, _ = transitions[t]

            # G_{t:t+n} 계산 (슬라이드 14페이지 공식)
            G = compute_n_step_return(transitions, t, n, V, gamma)

            # 갱신: V(St) ← V(St) + α[ G_{t:t+n} − V(St) ]
            td_error   = G - V[state]
            V[state]  += alpha * td_error
            ep_td_errors.append(abs(td_error))

        td_errors_history.append(np.mean(ep_td_errors))

        # ── 진행 상황 출력 ────────────────────────────────────────
        if (ep + 1) % PRINT_INTERVAL == 0:
            n_label = "TD(0)" if n == 1 else f"{n}-step TD"
            print(f"  [{n_label:<10}]  Episode {ep+1:>7} | "
                  f"α={alpha:.4f} | γ={gamma:.2f} | "
                  f"TD Error (Avg): {td_errors_history[-1]:.6f}")

    n_label = "TD(0)" if n == 1 else f"{n}-step TD"
    print(f"  → [{n_label}] 예측 완료\n")
    return V, td_errors_history


# ── 전체 비교 실행 ─────────────────────────────────────────────
def run_comparison(env, n_list=N_STEP_LIST,
                   n_episodes=N_EPISODES, alpha=ALPHA, gamma=GAMMA):
    """
    여러 n 값에 대해 n-스텝 TD 예측 수행 및 비교

    Returns:
        results (dict): {n: (V, td_errors_history)}
    """
    results = {}
    for n in n_list:
        n_label = "TD(0) ← n=1 (1-스텝)" if n == 1 else f"n={n} ({n}-스텝 TD)"
        print(f"[{n_label}]")
        V, td_errors_history = n_step_td_prediction(
            env, n, n_episodes, alpha, gamma)
        results[n] = (V, td_errors_history)
    return results


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(results):
    """수렴된 V(s) 격자 비교 출력"""
    print("\n" + "=" * 65)
    print("  수렴된 V(s) 비교: n=1 (TD(0)) vs n=2 vs n=4")
    print("  [슬라이드 16페이지: 다양한 n 값에 대한 n-스텝 TD 방법]")
    print("=" * 65)

    # 테이블 헤더
    header = f"  {'State':<10} {'Tile':>4}"
    for n in N_STEP_LIST:
        label = "n=1(TD0)" if n == 1 else f"n={n}     "
        header += f"  {label:>10}"
    print(header)
    print("  " + "-" * (16 + 12 * len(N_STEP_LIST)))

    for s in range(N_STATES):
        r, c  = divmod(s, WIDTH)
        tile  = get_tile(s)
        label = f"({r},{c})"
        row   = f"  {label:<10} {tile:>4}"
        for n in N_STEP_LIST:
            V, _ = results[n]
            if tile in ('H', 'G'):
                row += f"  {'terminal':>10}"
            else:
                row += f"  {V[s]:>10.6f}"
        print(row)

    # 격자 비교
    method_labels = {1: "n=1  TD(0)   [1-스텝 TD]",
                     2: "n=2  2-스텝 TD",
                     4: "n=4  4-스텝 TD"}
    for n in N_STEP_LIST:
        V, _ = results[n]
        print(f"\n  [{method_labels[n]}]")
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
def visualize(results):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "n-step TD Prediction  |  FrozenLake 4×4  |  "
        f"γ={GAMMA}  |  α={ALPHA}  |  Episodes={N_EPISODES}",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.65, wspace=0.40,
                           top=0.93, bottom=0.08)

    # 방법별 설정
    method_cfg = {
        1: dict(color='#1976D2', lw=2.5, ls='-',
                label='n=1  TD(0)',
                title="(A) n=1  TD(0)  V(s)\n"
                      "Target: R_{t+1} + γV(S_{t+1})\n"
                      "[1-step TD]"),
        2: dict(color='#43A047', lw=2.0, ls='-',
                label='n=2  2-step TD',
                title="(B) n=2  2-step TD  V(s)\n"
                      "Target: R_{t+1} + γR_{t+2} + γ²V(S_{t+2})\n"
                      "[2-step Return]"),
        4: dict(color='#E53935', lw=2.0, ls='-',
                label='n=4  4-step TD',
                title="(C) n=4  4-step TD  V(s)\n"
                      "Target: R_{t+1}+...+γ⁴V(S_{t+4})\n"
                      "[4-step Return]"),
    }
    colors_cell = {'H': '#37474F', 'G': '#1565C0'}

    # ── (A)(B)(C) V(s) 열지도 ─────────────────────────────────
    def draw_heatmap(ax, V, title):
        v_grid = np.zeros((HEIGHT, WIDTH))
        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            if get_tile(s) not in ('H', 'G'):
                v_grid[r, c] = V[s]
        vmax = max(V.max(), 0.01)
        im = ax.imshow(v_grid, cmap='YlOrRd', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            tile = get_tile(s)
            if tile in ('H', 'G'):
                ax.add_patch(plt.Rectangle(
                    (c - .5, r - .5), 1, 1,
                    facecolor=colors_cell[tile], zorder=2,
                    edgecolor='white', linewidth=2))
                ax.text(c, r, tile, ha='center', va='center',
                        fontsize=13, color='white',
                        fontweight='bold', zorder=3)
            else:
                norm_val = V[s] / vmax if vmax > 0 else 0
                txt_color = 'white' if norm_val > 0.55 else 'black'
                ax.text(c, r, f"{V[s]:.3f}", ha='center', va='center',
                        fontsize=9, color=txt_color,
                        fontweight='bold', zorder=3)
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(HEIGHT - 0.5, -0.5)
        ax.set_xticks(range(WIDTH))
        ax.set_xticklabels([f"col{c}" for c in range(WIDTH)], fontsize=8)
        ax.set_yticks(range(HEIGHT))
        ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_aspect('equal')

    for col_idx, n in enumerate(N_STEP_LIST):
        ax = fig.add_subplot(gs[0, col_idx])
        V, _ = results[n]
        draw_heatmap(ax, V, method_cfg[n]['title'])

    # ── (D) TD 오차 감소 비교 ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :])
    window = N_EPISODES // 100

    for n in N_STEP_LIST:
        _, td_errors = results[n]
        cfg = method_cfg[n]
        if len(td_errors) >= window:
            smoothed = np.convolve(
                td_errors, np.ones(window) / window, mode='valid')
        else:
            smoothed = td_errors
        ax_d.plot(smoothed, color=cfg['color'],
                  linewidth=cfg['lw'], linestyle=cfg['ls'],
                  label=cfg['label'], alpha=0.9)

    ax_d.set_xlabel("Episode", fontsize=10)
    ax_d.set_ylabel("|G_{t:t+n} − V(St)|  (TD Error)", fontsize=10)
    ax_d.set_title(
        f"(D) TD Error per Episode  (Moving Avg window={window})\n"
        "Larger n reflects more future rewards → Tends to increase variance",
        fontsize=10, fontweight='bold')
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.3)

    plt.savefig('./c_n_step_td_prediction_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] c_n_step_td_prediction_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  n-스텝 시간차 예측 (n-step TD Prediction)")
    print("  [FrozenLake 4×4  |  is_slippery=False]")
    print("  [갱신식: V(St) ← V(St) + α[ G_{t:t+n} − V(St) ]]")
    print("=" * 65)

    print(f"\n[비교 대상 n 값: {N_STEP_LIST}]")
    print(f"  n=1  →  TD(0)    : 1-스텝 TD  "
          f"(Bootstrap 1회, Time-step 업데이트)")
    print(f"  n=2  →  2-스텝 TD: 2칸 앞 보상까지 반영 후 Bootstrap")
    print(f"  n=4  →  4-스텝 TD: 4칸 앞 보상까지 반영 후 Bootstrap")
    print(f"\n  [슬라이드 16페이지]")
    print(f"  n=1:  V(St) ← V(St) + α[ R_{{t+1}} + γV(S_{{t+1}}) − V(St) ]")
    print(f"  n=2:  V(St) ← V(St) + α[ R_{{t+1}} + γR_{{t+2}} "
          f"+ γ²V(S_{{t+2}}) − V(St) ]")
    print(f"  n=4:  V(St) ← V(St) + α[ R_{{t+1}} + ... "
          f"+ γ⁴V(S_{{t+4}}) − V(St) ]\n")

    env = gym.make("FrozenLake-v1", desc=MAP_4x4,
                   is_slippery=False, render_mode=None)

    results = run_comparison(env)

    print_results(results)
    visualize(results)

    env.close()


if __name__ == '__main__':
    main()
