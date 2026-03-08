"""
Frozen Lake — 반복적 정책 평가 (Iterative Policy Evaluation)
결정적(is_slippery=False) vs. 확률적(is_slippery=True) 환경 비교

05.DP.pptx 슬라이드 3-5 대응

핵심 개념:
  - 결정적 환경:  p(s'|s,a) = 1
      v_π(s) = Σ_a π(a|s) * [r(s,a,s') + γ * v_π(s')]

  - 확률적 환경:  p(s'|s,a) ≠ 1  (is_slippery=True: 각 방향 1/3 확률)
      v_π(s) = Σ_a π(a|s) * Σ_{s'} p(s'|s,a) * [r + γ * v_π(s')]

  → 슬라이드에서의 "Bootstrap": v(s)를 계산할 때 다른 상태의 v(s')를 재사용
  → 벨만 기대 방정식을 반복 적용 → 수렴 (Contraction Mapping 보장)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════
# 1. 공통 설정
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]

GAMMA     = 0.99
PI        = 0.25     # 균등 랜덤 정책
HEIGHT    = 4
WIDTH     = 4
N_STATES  = HEIGHT * WIDTH
N_ACTIONS = 4        # LEFT=0, DOWN=1, RIGHT=2, UP=3

ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}

CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

# ══════════════════════════════════════════════════
# 2. 반복적 정책 평가 (Iterative Policy Evaluation)
#
#  결정적/확률적 모두 동일한 코드 사용:
#  v(s) ← Σ_a π(a|s) * Σ_{s'} p(s'|s,a) * [r + γ·v(s')]
#
#  gymnasium P[s][a] = [(prob, next_s, reward, terminated), ...]
#  결정적: prob=1.0 → Σ_{s'} p(...) = 단일 항
#  확률적: prob=1/3 × 3항 → 기댓값 계산
# ══════════════════════════════════════════════════
def iterative_policy_evaluation(env, gamma=GAMMA, theta=1e-9, max_iter=200000):
    """
    벨만 기대 방정식 반복 적용 — 결정적·확률적 환경 공통 함수

    핵심: env.unwrapped.P[s][a]의 prob을 그대로 사용하면
          결정적(prob=1.0)과 확률적(prob<1.0) 환경을 동일하게 처리 가능

    Returns:
        V       : shape (HEIGHT, WIDTH)  수렴된 상태 가치 함수
        delta_history : 각 반복의 최대 변화량 리스트
    """
    V = np.zeros(N_STATES)
    delta_history = []

    for iteration in range(max_iter):
        delta = 0.0
        V_new = np.copy(V)

        for s in range(N_STATES):
            row, col = divmod(s, WIDTH)
            # 종료 상태(H, G)는 가치 = 0 고정
            if CELL[(row, col)] in ('H', 'G'):
                V_new[s] = 0.0
                continue

            # 벨만 기대 방정식 한 번 적용 (one-step backup)
            #   v_new(s) = Σ_a π(a|s) * Σ_{s'} p(s'|s,a) * [r + γ·V(s')]
            v_new = 0.0
            for a in range(N_ACTIONS):
                for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                    # ↑ prob: 결정적=1.0 / 확률적=0.333...
                    v_new += PI * prob * (reward + gamma * V[next_s])

            delta    = max(delta, abs(V[s] - v_new))
            V_new[s] = v_new

        V = V_new
        delta_history.append(delta)

        if delta < theta:
            print(f"  수렴: {iteration+1}회 반복  (Δ={delta:.2e} < θ={theta:.0e})")
            break
    else:
        print(f"  최대 {max_iter}회 도달  (Δ={delta:.2e})")

    return V.reshape(HEIGHT, WIDTH), delta_history


def print_comparison(V_det, V_sto):
    """결정적 vs. 확률적 가치 함수 콘솔 비교"""
    print(f"\n{'='*65}")
    print(f"  반복적 정책 평가 결과  (γ={GAMMA}, 균등 랜덤 정책 π=0.25)")
    print(f"{'='*65}")
    header = f"  {'상태':^8}  {'결정적 V_π(s)':^14}  {'확률적 V_π(s)':^14}  {'차이':^10}"
    print(header)
    print("  " + "-" * 52)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            s_label = f"({r},{c})[{cell}]"
            if cell in ('H', 'G'):
                print(f"  {s_label:^8}  {'terminal':^14}  {'terminal':^14}  {'—':^10}")
            else:
                vd = V_det[r, c]
                vs = V_sto[r, c]
                diff = vs - vd
                print(f"  {s_label:^8}  {vd:^14.6f}  {vs:^14.6f}  {diff:^+10.6f}")


# ══════════════════════════════════════════════════
# 3. 시각화
# ══════════════════════════════════════════════════
def draw_heatmap(ax, V_grid, title, cmap_name='RdYlGn'):
    """Draw state value function heatmap"""
    V_plot = V_grid.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                V_plot[r, c] = np.nan

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad(color='#cccccc')
    vmax = np.nanmax(np.abs(V_plot)) + 1e-9
    im = ax.imshow(V_plot, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#37474F', zorder=2))
                ax.text(c, r, "H\n(Hole)", ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#1565C0', zorder=2))
                ax.text(c, r, "G\n(Goal)", ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=3)
            else:
                label_prefix = "S\n" if cell == 'S' else ""
                ax.text(c, r, f"{label_prefix}{V_grid[r,c]:.4f}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='black', zorder=3)

    ax.set_xticks(range(WIDTH))
    ax.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax.set_yticks(range(HEIGHT))
    ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.82, label="V_pi(s)")
    return im


def visualize(V_det, V_sto, delta_det, delta_sto):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Iterative Policy Evaluation\n"
        f"Deterministic (is_slippery=False) vs. Stochastic (is_slippery=True)  |  "
        f"Uniform Random Policy pi=0.25  |  gamma={GAMMA}",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── (A) Deterministic env value function ──────────
    ax_det = fig.add_subplot(gs[0, 0])
    draw_heatmap(ax_det, V_det,
        f"(A) Deterministic  V_pi(s)\n"
        f"is_slippery=False  |  {len(delta_det)} iters")

    # ── (B) Stochastic env value function ─────────────
    ax_sto = fig.add_subplot(gs[0, 1])
    draw_heatmap(ax_sto, V_sto,
        f"(B) Stochastic  V_pi(s)\n"
        f"is_slippery=True  |  {len(delta_sto)} iters",
        cmap_name='PuOr')

    # ── (C) Difference: Stochastic - Deterministic ────
    ax_diff = fig.add_subplot(gs[0, 2])
    diff = V_sto - V_det
    diff_plot = diff.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                diff_plot[r, c] = np.nan

    cmap_d = plt.cm.RdBu.copy()
    cmap_d.set_bad(color='#cccccc')
    vmax_d = np.nanmax(np.abs(diff_plot)) + 1e-9
    im_d = ax_diff.imshow(diff_plot, cmap=cmap_d,
                           vmin=-vmax_d, vmax=vmax_d, aspect='auto')
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax_diff.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#37474F', zorder=2))
                ax_diff.text(c, r, "H", ha='center', va='center',
                             fontsize=10, color='white', zorder=3)
            elif cell == 'G':
                ax_diff.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#1565C0', zorder=2))
                ax_diff.text(c, r, "G", ha='center', va='center',
                             fontsize=10, color='white', zorder=3)
            else:
                sign = "+" if diff[r,c] >= 0 else ""
                ax_diff.text(c, r, f"{sign}{diff[r,c]:.4f}",
                             ha='center', va='center', fontsize=10,
                             fontweight='bold', color='black', zorder=3)
    ax_diff.set_xticks(range(WIDTH))
    ax_diff.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_diff.set_yticks(range(HEIGHT))
    ax_diff.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_diff.set_title("(C) Difference  V_sto(s) - V_det(s)\n(Negative = Stochastic env is riskier)",
                      fontsize=11, fontweight='bold')
    plt.colorbar(im_d, ax=ax_diff, shrink=0.82, label="Delta V_pi(s)")

    # ── (D) Convergence curve comparison ─────────────
    ax_conv = fig.add_subplot(gs[1, :2])
    ax_conv.semilogy(delta_det, color='#1976D2', linewidth=2,
                     label=f'Deterministic  is_slippery=False  ({len(delta_det)} iters)')
    ax_conv.semilogy(delta_sto, color='#E53935', linewidth=2,
                     label=f'Stochastic     is_slippery=True   ({len(delta_sto)} iters)')
    ax_conv.axhline(1e-9, color='gray', linestyle='--', linewidth=1.2,
                    label='Convergence threshold  theta=1e-9')
    ax_conv.set_xlabel("Iteration", fontsize=11)
    ax_conv.set_ylabel("Max Delta  (log scale)", fontsize=11)
    ax_conv.set_title("(D) Convergence Comparison\nDeterministic vs. Stochastic Environment",
                      fontsize=12, fontweight='bold')
    ax_conv.legend(fontsize=10)
    ax_conv.grid(True, which='both', alpha=0.3)

    # ── (E) Bellman equation summary box ─────────────
    ax_eq = fig.add_subplot(gs[1, 2])
    ax_eq.axis('off')
    equations = (
        "Bellman Expectation Equation\n\n"
        "Deterministic  [p(s'|s,a) = 1]:\n"
        "  v_pi(s) = Sum_a pi(a|s)\n"
        "            x [r + g*v_pi(s')]\n\n"
        "Stochastic  [p(s'|s,a) <= 1]:\n"
        "  v_pi(s) = Sum_a pi(a|s)\n"
        "            x Sum_{s'} p(s'|s,a)\n"
        "            x [r + g*v_pi(s')]\n\n"
        "Iterative Update (Bootstrap):\n"
        "  v_{k+1}(s) <- Sum_a pi(a|s)\n"
        "    x Sum_{s'} p(s'|s,a)\n"
        "    x [r + g*v_k(s')]\n\n"
        "-> k->inf  =>  v_k -> v_pi"
    )
    ax_eq.text(0.05, 0.95, equations,
               transform=ax_eq.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#E3F2FD',
                         edgecolor='#1565C0', linewidth=1.5))
    ax_eq.set_title("(E) Key Equations Summary", fontsize=11, fontweight='bold')

    legend_elems = [
        mpatches.Patch(color='#37474F', label='H: Hole  (terminal, fail)'),
        mpatches.Patch(color='#1565C0', label='G: Goal  (terminal, success)'),
    ]

    plt.savefig("./a_policy_evaluation_stochastic_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: a_policy_evaluation_stochastic_img.png")


def main():
    # ── 두 환경 생성 ─────────────────────────────────
    env_det = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)
    env_sto = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=True,  render_mode=None)

    print("\n[1] 결정적 환경  (is_slippery=False)")
    V_det, delta_det = iterative_policy_evaluation(env_det)

    print("\n[2] 확률적 환경  (is_slippery=True)")
    V_sto, delta_sto = iterative_policy_evaluation(env_sto)

    # ── 콘솔 비교 출력 ────────────────────────────────
    print_comparison(V_det, V_sto)

    # ── 시각화 ────────────────────────────────────────
    visualize(V_det, V_sto, delta_det, delta_sto)

    env_det.close()
    env_sto.close()


if __name__ == "__main__":
    main()
