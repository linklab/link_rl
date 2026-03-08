"""
Frozen Lake — 정책 개선 (Policy Improvement)
정책 개선 정리(Policy Improvement Theorem) 시각화

05.DP.pptx 슬라이드 7-9 대응

핵심 개념:
  1) 정책 개선 (Policy Improvement)
     π'(s) = argmax_a  Σ_{s'} p(s'|s,a) · [r + γ · v_π(s)]
     ≡ argmax_a  q_π(s, a)

  2) 정책 개선 정리 (Policy Improvement Theorem)
     q_π(s, π'(s)) ≥ v_π(s)  for all s
     → V_π'(s) ≥ V_π(s)      for all s  (개선된 정책이 반드시 더 좋거나 같음)

  3) 시각화:
     (A) 랜덤 정책 π (균등: 0.25) 의 가치 함수 V_π
     (B) 랜덤 정책 기반 탐욕 정책 π' = greedy(V_π)
     (C) 개선된 정책 π' 의 가치 함수 V_π'
     (D) 개선 전·후 가치 함수 차이  V_π'(s) - V_π(s) ≥ 0
     (E) 정책 개선 정리 검증: q_π(s, π'(s)) vs v_π(s)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════
# 1. 환경 설정
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]

GAMMA     = 0.99
PI_RAND   = 0.25     # 균등 랜덤 정책
HEIGHT    = 4
WIDTH     = 4
N_STATES  = HEIGHT * WIDTH
N_ACTIONS = 4        # LEFT=0, DOWN=1, RIGHT=2, UP=3

ACTION_NAMES  = {0: "←", 1: "↓", 2: "→", 3: "↑"}
ACTION_LABELS = ["L", "D", "R", "U"]

CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ══════════════════════════════════════════════════
# 2. 반복적 정책 평가 (Iterative Policy Evaluation)
#    주어진 정책 policy(s) → V_π(s) 계산
# ══════════════════════════════════════════════════
def policy_evaluation(env, policy_grid, gamma=GAMMA, theta=1e-9, max_iter=200000):
    """
    policy_grid: shape (HEIGHT, WIDTH)  각 셀의 행동 인덱스 또는 None(랜덤 정책)
    - policy_grid is None  →  균등 랜덤 정책 (π=0.25)
    - policy_grid is ndarray →  결정적 정책 (선택 행동에만 확률 1.0)

    Returns V: shape (HEIGHT, WIDTH)
    """
    V = np.zeros(N_STATES)
    delta_hist = []

    for _ in range(max_iter):
        delta = 0.0
        V_new = np.copy(V)

        for s in range(N_STATES):
            row, col = divmod(s, WIDTH)
            if CELL[(row, col)] in ('H', 'G'):
                V_new[s] = 0.0
                continue

            v_new = 0.0
            for a in range(N_ACTIONS):
                if policy_grid is None:
                    # 균등 랜덤 정책
                    pi_a = PI_RAND
                else:
                    # 결정적 정책: 선택된 행동만 확률 1.0
                    pi_a = 1.0 if policy_grid[row, col] == a else 0.0

                for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                    v_new += pi_a * prob * (reward + gamma * V[ns])

            delta    = max(delta, abs(V[s] - v_new))
            V_new[s] = v_new

        V = V_new
        delta_hist.append(delta)

        if delta < theta:
            break

    return V.reshape(HEIGHT, WIDTH)


# ══════════════════════════════════════════════════
# 3. 정책 개선 (Policy Improvement)
#    π'(s) = argmax_a  q_π(s,a)
#           = argmax_a  Σ_{s'} p(s'|s,a)[r + γ·v_π(s')]
#
#  정책 개선 정리:
#    q_π(s, π'(s)) = max_a q_π(s,a) ≥ Σ_a π(a|s)·q_π(s,a) = v_π(s)
#    따라서 V_π'(s) ≥ V_π(s)  for all s
# ══════════════════════════════════════════════════
def compute_q_values(env, V_flat, gamma=GAMMA):
    """
    V_flat: shape (N_STATES,)
    Returns Q: shape (N_STATES, N_ACTIONS)
      Q(s,a) = Σ_{s'} p(s'|s,a) * [r + γ·V(s')]
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                Q[s, a] += prob * (reward + gamma * V_flat[ns])
    return Q


def policy_improvement(env, V_pi, gamma=GAMMA):
    """
    V_pi: shape (HEIGHT, WIDTH)
    Returns:
        policy_new: shape (HEIGHT, WIDTH)  개선된 결정적 정책 (행동 인덱스)
        Q         : shape (N_STATES, N_ACTIONS) q_π(s,a)
    """
    V_flat = V_pi.flatten()
    Q = compute_q_values(env, V_flat, gamma)

    policy_new = np.full((HEIGHT, WIDTH), -1, dtype=int)
    for s in range(N_STATES):
        row, col = divmod(s, WIDTH)
        if CELL[(row, col)] in ('H', 'G'):
            continue
        policy_new[row, col] = int(np.argmax(Q[s]))

    return policy_new, Q


# ══════════════════════════════════════════════════
# 4. 출력 함수
# ══════════════════════════════════════════════════
def print_policy(policy_grid, title):
    print(f"\n  {title}")
    print("  ┌──────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':   row_str += "  H "
            elif cell == 'G': row_str += "  G "
            elif policy_grid is None:
                row_str += "  ? "    # 랜덤 정책
            else:
                row_str += f"  {ACTION_NAMES[policy_grid[r,c]]} "
        print(row_str + " │")
    print("  └──────────────────┘")


def print_improvement_theorem(V_pi, V_pi_prime, Q, policy_new):
    """정책 개선 정리 검증: q_π(s,π'(s)) ≥ v_π(s)"""
    print(f"\n  정책 개선 정리 검증  [q_π(s,π'(s)) ≥ v_π(s)?]")
    print(f"  {'상태':^8}  {'π*(s)':^6}  {'v_π(s)':^10}  {'q_π(s,π*(s))':^14}  {'개선?':^6}  {'V_π*(s)':^10}")
    print("  " + "-" * 62)
    Q_2d = Q.reshape(HEIGHT, WIDTH, N_ACTIONS)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            s_label = f"({r},{c})[{cell}]"
            if cell in ('H', 'G'):
                print(f"  {s_label:^8}  {'—':^6}  {'terminal':^10}")
                continue
            v  = V_pi[r, c]
            vp = V_pi_prime[r, c]
            a  = policy_new[r, c]
            q  = Q_2d[r, c, a]
            ok = "✓" if q >= v - 1e-9 else "✗"
            print(f"  {s_label:^8}  {ACTION_NAMES[a]:^6}  {v:^10.6f}  "
                  f"{q:^14.6f}  {ok:^6}  {vp:^10.6f}")


# ══════════════════════════════════════════════════
# 5. 시각화
# ══════════════════════════════════════════════════
def draw_value_map(ax, V_grid, title, policy_grid=None, cmap_name='RdYlGn'):
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
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#1565C0', zorder=2))
                ax.text(c, r, "G\n(Goal)", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            else:
                # 가치 값 표시
                ax.text(c, r - 0.18, f"{V_grid[r,c]:.4f}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='black', zorder=3)
                # 정책 화살표 (있는 경우)
                if policy_grid is None:
                    ax.text(c, r + 0.22, "?", ha='center', va='center',
                            fontsize=13, color='#795548', fontweight='bold', zorder=4)
                else:
                    a = policy_grid[r, c]
                    ax.text(c, r + 0.22, ACTION_NAMES[a],
                            ha='center', va='center',
                            fontsize=14, color='#1565C0', fontweight='bold', zorder=4)

    ax.set_xticks(range(WIDTH));  ax.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax.set_yticks(range(HEIGHT)); ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.82, label="V(s)")


def visualize(V_pi, V_pi_prime, policy_new, Q):
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"Policy Improvement\n"
        f"Uniform Random Policy pi -> Greedy Policy pi'  |  is_slippery=False  |  gamma={GAMMA}",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    # ── (A) Random policy value function V_pi ────────
    ax_a = fig.add_subplot(gs[0, 0])
    draw_value_map(ax_a, V_pi,
        "(A) Random Policy  V_pi(s)\n"
        "pi = Uniform 0.25  (? = random)",
        policy_grid=None)

    # ── (B) Improved policy pi' ──────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    draw_value_map(ax_b, V_pi_prime,
        "(B) Improved Policy  V_pi'(s)\n"
        "pi' = greedy(V_pi)  (arrow = pi')",
        policy_grid=policy_new,
        cmap_name='YlGn')

    # ── (C) Improvement  V_pi'(s) - V_pi(s) ─────────
    ax_c = fig.add_subplot(gs[0, 2])
    diff = V_pi_prime - V_pi
    diff_plot = diff.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                diff_plot[r, c] = np.nan

    cmap_d = plt.cm.Blues.copy()
    cmap_d.set_bad(color='#cccccc')
    im_d = ax_c.imshow(diff_plot, cmap=cmap_d,
                        vmin=0, vmax=np.nanmax(diff_plot) + 1e-9, aspect='auto')
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax_c.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#37474F', zorder=2))
                ax_c.text(c, r, "H", ha='center', va='center',
                          fontsize=10, color='white', zorder=3)
            elif cell == 'G':
                ax_c.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#1565C0', zorder=2))
                ax_c.text(c, r, "G", ha='center', va='center',
                          fontsize=10, color='white', zorder=3)
            else:
                ax_c.text(c, r, f"+{diff[r,c]:.4f}",
                          ha='center', va='center', fontsize=10,
                          fontweight='bold', color='black', zorder=3)
    ax_c.set_xticks(range(WIDTH));  ax_c.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_c.set_yticks(range(HEIGHT)); ax_c.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_c.set_title("(C) Improvement  V_pi'(s) - V_pi(s)\n"
                   "(Policy Improvement Theorem: always >= 0)",
                   fontsize=11, fontweight='bold')
    plt.colorbar(im_d, ax=ax_c, shrink=0.82, label="Delta V(s) >= 0")

    # ── (D) q_pi(s,a) heatmap: Q-values per cell ─────
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.set_xlim(-0.5, WIDTH - 0.5)
    ax_d.set_ylim(HEIGHT - 0.5, -0.5)
    ax_d.set_aspect('equal')
    ax_d.set_title("(D) q_pi(s,a) — Q-values for all 4 actions per cell\n"
                   "(Red bold = argmax = improved policy pi'(s))",
                   fontsize=12, fontweight='bold')

    Q_2d = Q.reshape(HEIGHT, WIDTH, N_ACTIONS)
    offsets = {0: (-0.30, 0.0), 1: (0.0, 0.28), 2: (0.30, 0.0), 3: (0.0, -0.28)}
    arrows  = ["←", "↓", "→", "↑"]

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            bg = '#37474F' if cell=='H' else ('#1565C0' if cell=='G' else '#F9FBE7')
            ax_d.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                           facecolor=bg, zorder=1,
                           edgecolor='#aaa', linewidth=1.5))
            if cell == 'H':
                ax_d.text(c, r, "H\n(Hole)", ha='center', va='center',
                          fontsize=11, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax_d.text(c, r, "G\n(Goal)", ha='center', va='center',
                          fontsize=11, color='white', fontweight='bold', zorder=3)
            else:
                best_a = int(np.argmax(Q_2d[r, c]))
                for a_idx in range(N_ACTIONS):
                    ox, oy = offsets[a_idx]
                    is_best = (a_idx == best_a)
                    ax_d.text(c + ox, r + oy,
                              f"{arrows[a_idx]}{Q_2d[r,c,a_idx]:.3f}",
                              ha='center', va='center', fontsize=10,
                              fontweight='bold' if is_best else 'normal',
                              color='#B71C1C' if is_best else '#333',
                              zorder=3)

    ax_d.set_xticks(range(WIDTH));  ax_d.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_d.set_yticks(range(HEIGHT)); ax_d.set_yticklabels([f"row{r}" for r in range(HEIGHT)])

    # ── (E) 정책 개선 정리 설명 ─────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')

    # q_π(s, π'(s)) vs v_π(s) 차이 계산
    diffs = []
    Q_2d_check = Q.reshape(HEIGHT, WIDTH, N_ACTIONS)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                continue
            a    = policy_new[r, c]
            q_sa = Q_2d_check[r, c, a]
            v_s  = V_pi[r, c]
            diffs.append(q_sa - v_s)

    min_diff = min(diffs)
    avg_diff = np.mean(diffs)

    theorem_text = (
        "Policy Improvement Theorem\n\n"
        "pi'(s) = argmax_a  q_pi(s,a)\n\n"
        "-> q_pi(s, pi'(s)) >= v_pi(s)  all s\n"
        "-> V_pi'(s) >= V_pi(s)          all s\n\n"
        "Verification Result:\n"
        f"  min  q_pi - v_pi = {min_diff:+.6f}\n"
        f"  mean q_pi - v_pi = {avg_diff:+.6f}\n\n"
        "-> Improvement guaranteed\n"
        "   for all non-terminal states ✓\n\n"
        "Intuition:\n"
        "  v_pi(s) is a weighted average\n"
        "  over all actions, so argmax\n"
        "  is always >= the average"
    )
    ax_e.text(0.05, 0.97, theorem_text,
              transform=ax_e.transAxes,
              fontsize=9.5, verticalalignment='top',
              fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F5E9',
                        edgecolor='#2E7D32', linewidth=1.5))
    ax_e.set_title("(E) Policy Improvement Theorem Verification",
                   fontsize=11, fontweight='bold')

    plt.savefig("./b_policy_improvement_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: b_policy_improvement_img.png")


def main():
    # ── Step 1: 랜덤 정책 평가 (V_π) ─────────────────
    print("\n[Step 1] 균등 랜덤 정책 평가  V_π(s) ...")
    V_pi = policy_evaluation(env, policy_grid=None)

    # ── Step 2: 정책 개선 π' = greedy(V_π) ───────────
    print("\n[Step 2] 정책 개선  π'(s) = argmax_a q_π(s,a) ...")
    policy_new, Q = policy_improvement(env, V_pi)

    # ── Step 3: 개선된 정책 재평가 (V_π') ─────────────
    print("\n[Step 3] 개선된 정책 평가  V_π'(s) ...")
    V_pi_prime = policy_evaluation(env, policy_grid=policy_new)

    # ── 콘솔 출력 ─────────────────────────────────────
    print_policy(None, "랜덤 정책 π  (균등 0.25)")
    print_policy(policy_new, "개선된 정책 π'  (greedy w.r.t. V_π)")
    print_improvement_theorem(V_pi, V_pi_prime, Q, policy_new)

    # ── 시각화 ────────────────────────────────────────
    visualize(V_pi, V_pi_prime, policy_new, Q)

    env.close()


if __name__ == "__main__":
    main()
