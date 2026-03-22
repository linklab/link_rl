"""
배치 업데이트 시간차 예측 (Batch Update TD Prediction)
FrozenLake 4×4 환경 기반 | 온라인 TD(0) vs 배치 TD vs 배치 MC 비교

[강의 핵심: 배치(Batch) 정의 - 슬라이드 21페이지]
  배치 = 임의 횟수의 에피소드 수행으로부터 얻은 누적 경험 샘플
  One Sample = (s, a, r', s', done)
  → 여러 에피소드 → 여러 샘플 → 배치(Batch) 구성

[배치 업데이트 갱신 식 - 슬라이드 22페이지]
  상태 s에 대해 n개의 경험 샘플이 존재할 때:

  V(St) ← V(St) + α · [ (Σ_{i=1}^{n} R_{t+1,i} + γV(S_{t+1,i})) / n  −  V(St) ]

  → 배치 내 TD 타겟 값을 각 상태별로 모두 합치고 평균을 내어 한 번에 갱신
  → 해당 배치 내 경험 샘플을 반복적으로 사용 가능

[배치 TD 정확성 이유 3가지 - 슬라이드 26페이지]
  1) 여러 샘플을 한꺼번에 고려하여 타겟 값 추출
  2) 동일 샘플을 반복적으로 여러 번 활용하여 학습 가능
  3) 부트스트랩 방법 사용 → 다른 상태의 가치 정보를 활용하여 추정

[슬라이드 26페이지 비교표]
              배치 활용 X    배치 활용 O
  TD          V(A) = 0       V(A) = 3/4  ← 참 가치와 일치!
  MC          V(A) = 0       V(A) = 0    ← 참 가치와 불일치
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA        = 0.99   # 감가율
N_EPISODES   = 3_000  # 총 에피소드 수
ALPHA        = 0.01   # 학습률
BATCH_REPEAT = 3      # 배치 내 샘플 반복 학습 횟수

PRINT_INTERVAL = 500  # 콘솔 출력 간격

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
    """균등 랜덤 정책: 4개 방향 중 무작위 선택"""
    return np.random.randint(N_ACTIONS)


# ══════════════════════════════════════════════════════════════
# [방법 1] 온라인(Online) TD(0) - 배치 미사용
# ══════════════════════════════════════════════════════════════
def online_td_prediction(env, n_episodes=N_EPISODES,
                          alpha=ALPHA, gamma=GAMMA):
    """
    온라인 TD(0) 예측 (배치 미사용) - 슬라이드 7페이지 의사코드 구현

    매 타임 스텝마다 즉시 V(s) 갱신:
    V(St) ← V(St) + α[ R_{t+1} + γV(S_{t+1}) − V(St) ]

    [슬라이드 25페이지: 배치 사용하지 않을 때]
    - V(A)를 업데이트할 때 V(B) = 0 → V(A) = 0

    Returns:
        V                 (np.ndarray): 학습된 V(s)  [N_STATES]
        td_errors_history (list):       에피소드별 평균 TD 오차
    """
    V = np.zeros(N_STATES)
    td_errors_history = []

    for ep in range(n_episodes):
        # ── 슬라이드 의사코드 5행: Initialize s ───────────────
        state, _ = env.reset()
        ep_td_errors = []

        # ── 슬라이드 의사코드 6~15행: for each time step ──────
        while True:
            action = policy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD 타겟 계산 (슬라이드 9~12행)
            if terminated:
                td_target = reward                           # if s' is terminal
            else:
                td_target = reward + gamma * V[next_state]  # else: TD 타겟

            td_error  = td_target - V[state]   # δt (TD 오차)
            V[state] += alpha * td_error        # 즉시 갱신 (온라인)

            ep_td_errors.append(abs(td_error))
            state = next_state
            if terminated or truncated:
                break

        td_errors_history.append(np.mean(ep_td_errors))

        if (ep + 1) % PRINT_INTERVAL == 0:
            print(f"  [온라인 TD]  Episode {ep+1:>5} | "
                  f"TD Error (Avg): {td_errors_history[-1]:.6f}")

    print(f"  → 온라인 TD(0) 완료\n")
    return V, td_errors_history


# ══════════════════════════════════════════════════════════════
# [방법 2] 배치 업데이트 TD - 슬라이드 22페이지 갱신 식 구현
# ══════════════════════════════════════════════════════════════
def batch_td_prediction(env, n_episodes=N_EPISODES,
                         alpha=ALPHA, gamma=GAMMA,
                         batch_repeat=BATCH_REPEAT):
    """
    배치 업데이트 TD 예측 (슬라이드 21~22페이지)

    [동작 방식]
    - 에피소드가 끝날 때마다 지금까지 수집된 모든 샘플을 배치로 활용
    - 배치 내 각 상태에 대해 TD 타겟 평균으로 한 번에 갱신

    [갱신 식 - 슬라이드 22페이지]
    V(St) ← V(St) + α · [ (Σ R_{t+1,i} + γV(S_{t+1,i})) / n  −  V(St) ]

    [슬라이드 25페이지: 배치 사용할 때]
    - 배치를 활용하면 V(B) ≈ 3/4 추정 후 → V(A) ≈ 3/4 (참 가치 일치)

    Returns:
        V                 (np.ndarray): 학습된 V(s)
        td_errors_history (list):       에피소드별 배치 갱신 후 평균 TD 오차
    """
    V = np.zeros(N_STATES)

    # 배치: 누적 경험 샘플  list of (s, r, s', done)
    # [슬라이드 21페이지: One Sample = (s, a', r', s'', done)]
    batch = []
    td_errors_history = []

    for ep in range(n_episodes):
        # ── 1. 에피소드 수행 & 배치에 샘플 누적 ─────────────────
        # [슬라이드 20~21페이지: Trajectory → Samples → Batch]
        state, _ = env.reset()
        while True:
            action = policy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 샘플 (s, r, s', done) 배치에 추가
            batch.append((state, reward, next_state, terminated))

            state = next_state
            if terminated or truncated:
                break

        # ── 2. 배치 업데이트 (슬라이드 22페이지 갱신 식) ─────────
        #    배치 내 샘플을 batch_repeat 번 반복 사용
        ep_td_errors = []
        for _ in range(batch_repeat):
            # 상태별 TD 타겟 누적 (분자: Σ R_{t+1} + γV(S_{t+1}))
            td_target_sum = np.zeros(N_STATES)
            td_target_cnt = np.zeros(N_STATES, dtype=int)

            for (s, r, s_next, done) in batch:
                if done:
                    td_target = r                          # 종료: 부트스트랩 없음
                else:
                    td_target = r + gamma * V[s_next]     # TD 타겟 (부트스트랩)
                td_target_sum[s] += td_target
                td_target_cnt[s] += 1

            # 각 상태별 평균 TD 타겟으로 V(s) 한 번에 갱신
            # [슬라이드 22페이지] V(St) ← V(St) + α·[Σ(TD target)/n − V(St)]
            for s in range(N_STATES):
                if td_target_cnt[s] > 0:
                    avg_td_target = td_target_sum[s] / td_target_cnt[s]
                    delta = avg_td_target - V[s]
                    V[s] += alpha * delta
                    ep_td_errors.append(abs(delta))

        td_errors_history.append(np.mean(ep_td_errors) if ep_td_errors else 0.0)

        if (ep + 1) % PRINT_INTERVAL == 0:
            print(f"  [배치 TD]    Episode {ep+1:>5} | 배치 크기: {len(batch):>6} | "
                  f"TD Error (Avg): {td_errors_history[-1]:.6f}")

    print(f"  → 배치 TD 완료  (최종 배치 크기: {len(batch)} 샘플)\n")
    return V, td_errors_history


# ══════════════════════════════════════════════════════════════
# [방법 3] 배치 업데이트 MC - 비교 기준
# ══════════════════════════════════════════════════════════════
def batch_mc_prediction(env, n_episodes=N_EPISODES,
                         alpha=ALPHA, gamma=GAMMA,
                         batch_repeat=BATCH_REPEAT):
    """
    배치 업데이트 MC 예측 (비교 기준)

    [슬라이드 24페이지: MC는 부트스트랩 없음]
    - V(B)를 추정할 때 V(A)를 활용하지 않음
    - 배치를 사용해도 MC는 실제 이득(Return) 샘플만 사용
    - 따라서 V(A) = 0 (참 가치 3/4와 불일치)

    Returns:
        V                 (np.ndarray): 학습된 V(s)
        mc_errors_history (list):       에피소드별 평균 MC 오차
    """
    V = np.zeros(N_STATES)

    # 배치: {상태: [이득 G1, G2, ...]}  (부트스트랩 없는 실제 이득)
    batch_returns = {}
    mc_errors_history = []

    for ep in range(n_episodes):
        # ── 1. 에피소드 수행: 실제 이득(Return) 계산 (부트스트랩 없음) ──
        state, _ = env.reset()
        episode_transitions = []
        while True:
            action = policy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_transitions.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        # MC: 역방향으로 실제 누적 이득 G 계산
        # [슬라이드 24페이지: 부트스트랩 사용 안 함 → V(B) 정보 활용 없음]
        G = 0.0
        for (s, r) in reversed(episode_transitions):
            G = gamma * G + r
            if s not in batch_returns:
                batch_returns[s] = []
            batch_returns[s].append(G)

        # ── 2. 배치 MC 갱신: 수집된 이득의 평균 → V(s) 갱신 ─────────
        ep_mc_errors = []
        for _ in range(batch_repeat):
            for s, returns in batch_returns.items():
                avg_return = np.mean(returns)
                delta = avg_return - V[s]
                V[s] += alpha * delta
                ep_mc_errors.append(abs(delta))

        mc_errors_history.append(np.mean(ep_mc_errors) if ep_mc_errors else 0.0)

        if (ep + 1) % PRINT_INTERVAL == 0:
            print(f"  [배치 MC]    Episode {ep+1:>5} | 방문 상태 수: {len(batch_returns):>3} | "
                  f"MC Error (Avg): {mc_errors_history[-1]:.6f}")

    print(f"  → 배치 MC 완료\n")
    return V, mc_errors_history


# ── 콘솔 출력 ─────────────────────────────────────────────────
def print_results(V_online_td, V_batch_td, V_batch_mc):
    """
    슬라이드 26페이지 표 형식으로 비교 출력
    """
    print("\n" + "=" * 65)
    print("  수렴된 V(s) 비교: 온라인 TD vs 배치 TD vs 배치 MC")
    print("  [슬라이드 26페이지: TD+배치 → MDP 참 가치에 가장 근접]")
    print("=" * 65)
    print(f"  {'State':<10} {'Tile':>4}  {'온라인 TD':>10}  {'배치 TD':>10}  {'배치 MC':>10}")
    print("  " + "-" * 52)
    for s in range(N_STATES):
        r, c  = divmod(s, WIDTH)
        tile  = get_tile(s)
        label = f"({r},{c})"
        if tile in ('H', 'G'):
            print(f"  {label:<10} {tile:>4}  {'terminal':>10}  {'terminal':>10}  {'terminal':>10}")
        else:
            print(f"  {label:<10} {tile:>4}  "
                  f"{V_online_td[s]:>10.6f}  "
                  f"{V_batch_td[s]:>10.6f}  "
                  f"{V_batch_mc[s]:>10.6f}")

    print("\n  V(s) 격자 비교:")
    methods = [("온라인 TD(0)", V_online_td),
               ("배치 TD     ", V_batch_td),
               ("배치 MC     ", V_batch_mc)]
    for name, V in methods:
        print(f"\n  [{name}]")
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
def visualize(V_online_td, V_batch_td, V_batch_mc,
              td_err_online, td_err_batch, mc_err_batch):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Batch Update TD Prediction Comparison\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  α={ALPHA}  |  "
        f"Episodes={N_EPISODES}  |  Batch Repeat={BATCH_REPEAT}",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                           top=0.88, bottom=0.08)

    colors_cell = {'H': '#37474F', 'G': '#1565C0'}

    def draw_heatmap(ax, V, title):
        """V(s) 열지도 공통 함수"""
        v_grid = np.zeros((HEIGHT, WIDTH))
        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            if get_tile(s) not in ('H', 'G'):
                v_grid[r, c] = V[s]
        vmax = max(np.max(np.abs(V)), 0.01)
        im = ax.imshow(v_grid, cmap='YlOrRd', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            tile = get_tile(s)
            if tile in ('H', 'G'):
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1,
                             facecolor=colors_cell[tile], zorder=2,
                             edgecolor='white', linewidth=2))
                ax.text(c, r, tile, ha='center', va='center',
                        fontsize=13, color='white', fontweight='bold', zorder=3)
            else:
                norm_val = V[s] / vmax if vmax > 0 else 0
                txt_color = 'white' if norm_val > 0.55 else 'black'
                ax.text(c, r, f"{V[s]:.3f}", ha='center', va='center',
                        fontsize=9, color=txt_color, fontweight='bold', zorder=3)
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(HEIGHT - 0.5, -0.5)
        ax.set_xticks(range(WIDTH))
        ax.set_xticklabels([f"col{c}" for c in range(WIDTH)], fontsize=8)
        ax.set_yticks(range(HEIGHT))
        ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_aspect('equal')

    # ── (A) 온라인 TD V(s) ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    draw_heatmap(ax_a, V_online_td,
                 "(A) Online TD(0)  V(s)\n"
                 "No Batch | Update per Step")

    # ── (B) 배치 TD V(s) ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    draw_heatmap(ax_b, V_batch_td,
                 f"(B) Batch TD  V(s)\n"
                 f"Batch Used | repeat={BATCH_REPEAT}")

    # ── (C) 배치 MC V(s) ────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    draw_heatmap(ax_c, V_batch_mc,
                 f"(C) Batch MC  V(s)\n"
                 f"Batch Used | repeat={BATCH_REPEAT} | No Bootstrap")

    # ── (D) 갱신 오차 감소 비교 ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0:2])
    window = max(N_EPISODES // 20, 10)

    def smooth(x):
        x = np.array(x)
        if len(x) >= window:
            return np.convolve(x, np.ones(window) / window, mode='valid')
        return x

    ax_d.plot(smooth(td_err_online), color='#43A047', linewidth=2.0,
              linestyle='--', label='Online TD(0)  (No Batch)', alpha=0.9)
    ax_d.plot(smooth(td_err_batch),  color='#1976D2', linewidth=2.2,
              label=f'Batch TD  (repeat={BATCH_REPEAT})', alpha=0.9)
    ax_d.plot(smooth(mc_err_batch),  color='#E53935', linewidth=1.8,
              linestyle=':', label=f'Batch MC  (repeat={BATCH_REPEAT})', alpha=0.9)
    ax_d.set_xlabel("Episode", fontsize=10)
    ax_d.set_ylabel("|δ|  (Update Error)", fontsize=10)
    ax_d.set_title(
        f"(D) Update Error per Episode  (Moving Avg window={window})\n"
        "Batch TD → Sample Reuse + Bootstrap → Convergence",
        fontsize=10, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.grid(True, alpha=0.3)

    # ── (E) 슬라이드 26페이지 원리 및 비교표 ─────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')

    # 원리 설명 텍스트
    lines = [
        ("Batch Update Rule",              0.95, 10, True,  '#1A237E'),
        ("V(St) ← V(St) + α ·",           0.85, 9,  False, '#37474F'),
        ("[ Σ(R+γV(S'))/n − V(St) ]",     0.77, 9,  False, '#1976D2'),
        ("",                                0.70, 1,  False, 'black'),
        ("Why Batch TD is Accurate",       0.63, 10, True,  '#1A237E'),
        ("① Aggregate targets from",      0.55, 9,  False, '#37474F'),
        ("   multiple samples at once",    0.48, 9,  False, '#37474F'),
        ("② Reuse same samples",          0.41, 9,  False, '#37474F'),
        ("   repeatedly for learning",     0.34, 9,  False, '#37474F'),
        ("③ Bootstrap propagates V",      0.27, 9,  False, '#37474F'),
        ("",                                0.20, 1,  False, 'black'),
        ("→ TD+Batch = MDP Solution",     0.13, 9,  True,  '#E53935'),
        ("  Most Accurate Convergence",    0.06, 9,  True,  '#E53935'),
    ]
    for (text, y, fs, bold, color) in lines:
        fw = 'bold' if bold else 'normal'
        ax_e.text(0.03, y, text, ha='left', va='center',
                  fontsize=fs, fontweight=fw, color=color,
                  transform=ax_e.transAxes)

    # 슬라이드 26 비교표 (우측 상단에 작게)
    cells  = [["Method", "No Batch", "Batch"],
               ["TD",  "V=0",  "V=3/4★"],
               ["MC",  "V=0",  "V=0"]]
    row_bg = ['#37474F', '#E3F2FD', '#FFF3E0']
    row_fg = ['white',   'black',   'black']
    tx = [0.55, 0.70, 0.85]
    ty = [0.95, 0.86, 0.77]
    for ri, row in enumerate(cells):
        for ci, cell in enumerate(row):
            ax_e.text(tx[ci], ty[ri], cell,
                      ha='center', va='center', fontsize=8,
                      fontweight='bold' if ri == 0 else 'normal',
                      color=row_fg[ri],
                      bbox=dict(boxstyle='round,pad=0.25',
                                facecolor=row_bg[ri],
                                edgecolor='white', linewidth=1),
                      transform=ax_e.transAxes)

    ax_e.set_title("(E) Batch Update Principle",
                   fontsize=10, fontweight='bold')

    plt.savefig('./c_batch_td_prediction_img.png', dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] c_batch_td_prediction_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  배치 업데이트 시간차 예측 (Batch Update TD Prediction)")
    print("  [FrozenLake 4×4  |  is_slippery=False]")
    print("  [갱신: V(St) ← V(St) + α·[Σ(TD target)/n − V(St)]]")
    print("=" * 65)

    env = gym.make("FrozenLake-v1", desc=MAP_4x4,
                   is_slippery=False, render_mode=None)

    # ── [방법 1] 온라인 TD(0) ──────────────────────────────────
    print("\n[방법 1] 온라인 TD(0)  (배치 미사용, 매 스텝 즉시 갱신)")
    print("  → 슬라이드 25페이지: V(A)를 업데이트할 때 V(B)=0 → V(A)=0")
    V_online_td, td_err_online = online_td_prediction(env)

    # ── [방법 2] 배치 업데이트 TD ──────────────────────────────
    print(f"[방법 2] 배치 업데이트 TD  (배치 사용, repeat={BATCH_REPEAT})")
    print("  → 슬라이드 25페이지: 배치로 V(B)≈3/4 추정 후 → V(A)≈3/4")
    V_batch_td, td_err_batch = batch_td_prediction(env)

    # ── [방법 3] 배치 업데이트 MC ──────────────────────────────
    print(f"[방법 3] 배치 업데이트 MC  (배치 사용, 부트스트랩 없음)")
    print("  → 슬라이드 24페이지: 배치 사용해도 MC는 V(A)=0 (참 가치 불일치)")
    V_batch_mc, mc_err_batch = batch_mc_prediction(env)

    print_results(V_online_td, V_batch_td, V_batch_mc)
    visualize(V_online_td, V_batch_td, V_batch_mc,
              td_err_online, td_err_batch, mc_err_batch)

    env.close()


if __name__ == '__main__':
    main()
