"""
Frozen Lake (is_slippery=False) — 상태 가치 함수 계산
벨만 기대 방정식 반복 계산 (Iterative Policy Evaluation)

환경 설정:
  - 4×4 Frozen Lake
  - is_slippery = False  (결정적 환경)
  - 균등 랜덤 정책: π(a|s) = 0.25 (4방향 동일)
  - 감가율 γ = 0.99

벨만 기대 방정식 (결정적 환경):
  v_π(s) = Σ_a π(a|s) * [r(s,a,s') + γ * v_π(s')]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ══════════════════════════════════════════════════
# 1. 환경 설정
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]

GAMMA   = 0.99   # 감가율
THETA   = 1e-9   # 수렴 기준
PI      = 0.25   # 균등 랜덤 정책
HEIGHT  = 4
WIDTH   = 4
N_STATES  = HEIGHT * WIDTH   # 16
N_ACTIONS = 4                # LEFT=0, DOWN=1, RIGHT=2, UP=3

ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# 셀 타입 맵
CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

HOLE_STATES = [(r, c) for (r, c), t in CELL.items() if t == 'H']
GOAL_STATES = [(r, c) for (r, c), t in CELL.items() if t == 'G']

env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ══════════════════════════════════════════════════
# 2. 벨만 기대 방정식 (Bellman Expectation)
#
#  v(s) ← Σ_a π(a|s) * Σ_{s'} p(s'|s,a) * [r + γ·v(s')]
#  결정적 환경이므로:
#  v(s) ← Σ_a π(a|s) * [r(s,a) + γ·v(s')]
# ══════════════════════════════════════════════════
def bellman_expectation_one_step(env, V, gamma=GAMMA):
    """
    벨만 기대 방정식을 모든 상태에 대해 한 번 적용 (one sweep)

    Returns:
        V_new (np.ndarray): 갱신된 가치 함수 (flat, N_STATES)
        delta (float):      이번 스윕의 최대 변화량
    """
    delta = 0.0
    V_new = np.copy(V)

    for s in range(N_STATES):
        row, col = divmod(s, WIDTH)

        # 종료 상태(H, G)는 가치 = 0 고정
        if CELL[(row, col)] in ('H', 'G'):
            V_new[s] = 0.0
            continue

        v_new = 0.0
        for a in range(N_ACTIONS):
            # p(s'|s,a), r, done
            for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                v_new += PI * prob * (reward + gamma * V[next_s])

        delta    = max(delta, abs(V[s] - v_new))
        V_new[s] = v_new

    return V_new, delta


def bellman_expectation(env, gamma=GAMMA, theta=THETA, max_iter=100000):
    """
    gymnasium env.unwrapped.P 를 직접 활용하여
    벨만 기대 방정식을 반복 적용

    P[s][a] = [(prob, next_state, reward, terminated), ...]
    """
    V = np.zeros(N_STATES)
    delta_history = []

    for iteration in range(max_iter):
        V, delta = bellman_expectation_one_step(env, V, gamma)
        delta_history.append(delta)

        if delta < theta:
            print(f"\n수렴 완료: {iteration+1}번째 반복  "
                  f"(Δ={delta:.2e} < θ={theta:.0e})")
            break
    else:
        print(f"최대 반복 {max_iter}회 도달  (최종 Δ={delta:.2e})")

    return V.reshape(HEIGHT, WIDTH), delta_history


def print_value_function(V_grid):
    print(f"\n▶ 수렴된 가치 함수 V_π(s)  (γ={GAMMA}, 균등 랜덤 정책)")
    print("       col0     col1     col2     col3")
    for r in range(HEIGHT):
        row_str = f"  row{r} "
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell in ('H', 'G'):
                row_str += f"  [{cell}]   "
            else:
                row_str += f" {V_grid[r, c]:7.4f}"
        print(row_str)


def visualize(env, V_grid, delta_hist):
    # ══════════════════════════════════════════════════
    # 3. 시각화
    # ══════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Frozen Lake 4×4  (is_slippery=False)\n"
                 f"Uniform Random Policy  π=0.25 | Discount Factor γ={GAMMA} | Bellman Expectation Equation",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ────────────────────────────────────────────────
    # (A) 가치 함수 히트맵
    # ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    # 히트맵용 배열 (H·G 셀은 NaN → 별도 색상)
    V_plot = V_grid.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                V_plot[r, c] = np.nan

    cmap_main = plt.cm.RdYlGn.copy()
    cmap_main.set_bad(color='#cccccc')   # NaN 셀 회색

    vmax = np.nanmax(np.abs(V_plot)) + THETA
    im = ax1.imshow(V_plot, cmap=cmap_main,
                    vmin=-vmax, vmax=vmax, aspect='auto')

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]

            if cell == 'H':
                ax1.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color='#37474F', zorder=2))
                ax1.text(c, r, "H\n(Hole)", ha='center', va='center',
                         fontsize=10, color='white', fontweight='bold', zorder=3)

            elif cell == 'G':
                ax1.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color='#1565C0', zorder=2))
                ax1.text(c, r, "G\n(Goal)", ha='center', va='center',
                         fontsize=10, color='white', fontweight='bold', zorder=3)

            elif cell == 'S':
                ax1.text(c, r - 0.28, "S", ha='center', va='center',
                         fontsize=8, color='#555', zorder=3)
                ax1.text(c, r + 0.15, f"{V_grid[r,c]:.4f}",
                         ha='center', va='center', fontsize=11,
                         fontweight='bold', color='black', zorder=3)
            else:
                ax1.text(c, r, f"{V_grid[r,c]:.4f}",
                         ha='center', va='center', fontsize=11,
                         fontweight='bold', color='black', zorder=3)

    ax1.set_xticks(range(WIDTH));  ax1.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax1.set_yticks(range(HEIGHT)); ax1.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax1.set_title(f"(A) State Value Function V_π(s)\n(γ={GAMMA})", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, shrink=0.85, label="V_π(s)")

    legend_elems = [
        mpatches.Patch(color='#37474F', label='H: Hole  (terminal, V=0)'),
        mpatches.Patch(color='#1565C0', label='G: Goal  (terminal, V=0)'),
    ]
    ax1.legend(handles=legend_elems, loc='upper right', fontsize=8)

    # ────────────────────────────────────────────────
    # (B) 수렴 곡선
    # ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(delta_hist, color='#1976D2', linewidth=2)
    ax2.axhline(THETA, color='red', linestyle='--', linewidth=1.2,
                label='θ = 1e-9 (convergence threshold)')
    ax2.axvline(len(delta_hist)-1, color='green', linestyle=':',
                linewidth=1.5, label=f'Converged ({len(delta_hist)} iters)')
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Max Delta  Δ  (log scale)", fontsize=11)
    ax2.set_title(f"(B) Convergence Curve\n(Total {len(delta_hist)} iterations)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    # ────────────────────────────────────────────────
    # (C) γ 값 변화에 따른 가치 함수 비교 (히트맵 × 4)
    # ────────────────────────────────────────────────
    gamma_list = [0.5, 0.9, 0.99, 1.0]
    ax_gamma = [fig.add_subplot(gs[1, 0]),
                None, None, None]

    # 2×2 서브그리드
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], wspace=0.35)

    axes_g = [fig.add_subplot(inner_gs[k]) for k in range(4)]

    for ax_g, g in zip(axes_g, gamma_list):
        V_g, _ = bellman_expectation(env, gamma=g, theta=THETA, max_iter=100000)
        V_g_plot = V_g.copy().astype(float)
        for r in range(HEIGHT):
            for c in range(WIDTH):
                if CELL[(r, c)] in ('H', 'G'):
                    V_g_plot[r, c] = np.nan

        vmax_g = np.nanmax(np.abs(V_g_plot)) + THETA
        cmap_g = plt.cm.RdYlGn.copy()
        cmap_g.set_bad(color='#cccccc')
        ax_g.imshow(V_g_plot, cmap=cmap_g,
                    vmin=-vmax_g, vmax=vmax_g, aspect='auto')

        for r in range(HEIGHT):
            for c in range(WIDTH):
                cell = CELL[(r, c)]
                if cell == 'H':
                    ax_g.add_patch(plt.Rectangle(
                        (c-0.5, r-0.5), 1, 1, color='#37474F', zorder=2))
                    ax_g.text(c, r, "H", ha='center', va='center',
                              fontsize=9, color='white', zorder=3)
                elif cell == 'G':
                    ax_g.add_patch(plt.Rectangle(
                        (c-0.5, r-0.5), 1, 1, color='#1565C0', zorder=2))
                    ax_g.text(c, r, "G", ha='center', va='center',
                              fontsize=9, color='white', zorder=3)
                else:
                    ax_g.text(c, r, f"{V_g[r,c]:.3f}",
                              ha='center', va='center',
                              fontsize=8, fontweight='bold', color='black', zorder=3)

        ax_g.set_xticks(range(WIDTH))
        ax_g.set_xticklabels([f"c{c}" for c in range(WIDTH)], fontsize=7)
        ax_g.set_yticks(range(HEIGHT))
        ax_g.set_yticklabels([f"r{r}" for r in range(HEIGHT)], fontsize=7)
        ax_g.set_title(f"(C) γ = {g}", fontsize=11, fontweight='bold')

    # ── 원래 ax_gamma[0] 제거 (inner_gs로 대체됨) ──
    ax_gamma[0].remove()

    plt.savefig("./a_state_values_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: a_state_values_img.png")


def main():
    # ── 계산 실행 ────────────────────────────────────
    V_grid, delta_hist = bellman_expectation(env, gamma=GAMMA)

    # ── 콘솔 출력 ────────────────────────────────────
    print_value_function(V_grid)

    # ── 시각화 ───────────────────────────────────────
    visualize(env, V_grid, delta_hist)

    env.close()


if __name__ == "__main__":
    main()