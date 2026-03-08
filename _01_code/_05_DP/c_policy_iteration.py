"""
Frozen Lake — 정책 반복 (Policy Iteration)
Policy Evaluation(E) + Policy Improvement(I) 반복 루프 시각화

05.DP.pptx 슬라이드 10-15 대응

핵심 개념:
  정책 반복 (Policy Iteration):
    1) 초기 임의 정책 π_0 설정
    2) 정책 평가 (E): V_π_k 계산 (벨만 기대 방정식 반복)
    3) 정책 개선 (I): π_{k+1} = greedy(V_π_k)
    4) π_{k+1} = π_k 이면 수렴 → 최적 정책 π* 도출
    5) 2-4 반복

  의사 코드 (슬라이드 13):
    초기화: V(s)=0, π(s)=random ∀s
    루프:
      # Policy Evaluation
      Δ ← 0
      while Δ < θ:
          for each s:
              v ← V(s)
              V(s) ← Σ_a π(a|s) Σ_{s'} p(s'|s,a)[r + γV(s')]
              Δ ← max(Δ, |v - V(s)|)
      # Policy Improvement
      policy_stable ← True
      for each s:
          old_action ← π(s)
          π(s) ← argmax_a Σ_{s'} p(s'|s,a)[r + γV(s')]
          if old_action ≠ π(s): policy_stable ← False
      if policy_stable: return V, π
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
HEIGHT    = 4
WIDTH     = 4
N_STATES  = HEIGHT * WIDTH
N_ACTIONS = 4        # LEFT=0, DOWN=1, RIGHT=2, UP=3

ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}

CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)


# ══════════════════════════════════════════════════
# 2. 정책 반복 (Policy Iteration)
#
#  내부 루프:
#    E-step: 정책 평가 (Iterative Policy Evaluation)
#    I-step: 정책 개선 (Greedy Policy Improvement)
# ══════════════════════════════════════════════════
def policy_evaluation_step(env, V, policy, gamma=GAMMA, theta=1e-6):
    """
    E-step: 주어진 결정적 정책 policy 에 대해 V를 수렴시킴
    policy: shape (N_STATES,)  행동 인덱스

    Returns:
        V (np.ndarray): 수렴된 가치 함수 (flat, N_STATES)
        n_sweeps (int): 수렴까지 걸린 스윕 수
    """
    n_sweeps = 0
    while True:
        delta = 0.0
        V_new = np.copy(V)
        for s in range(N_STATES):
            row, col = divmod(s, WIDTH)
            if CELL[(row, col)] in ('H', 'G'):
                V_new[s] = 0.0
                continue
            a = policy[s]
            v_new = 0.0
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                v_new += prob * (reward + gamma * V[ns])
            delta    = max(delta, abs(V[s] - v_new))
            V_new[s] = v_new
        V = V_new
        n_sweeps += 1
        if delta < theta:
            break
    return V, n_sweeps


def policy_improvement_step(env, V, gamma=GAMMA):
    """
    I-step: V에 대해 탐욕 정책 산출
    Returns:
        policy_new (np.ndarray): shape (N_STATES,)
        changed    (bool): 정책이 바뀌었는지 여부 (이전 인자 없이 단독 호출용)
    """
    policy_new = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        row, col = divmod(s, WIDTH)
        if CELL[(row, col)] in ('H', 'G'):
            policy_new[s] = 0
            continue
        q_vals = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                q_vals[a] += prob * (reward + gamma * V[ns])
        policy_new[s] = int(np.argmax(q_vals))
    return policy_new


def policy_iteration(env, gamma=GAMMA, theta=1e-6, max_outer=100):
    """
    완전한 정책 반복 (Policy Iteration)
    E-step → I-step → E-step → I-step → ... until policy_stable

    Returns:
        V_history      : 각 정책 반복 후 가치 함수 스냅샷  list of (HEIGHT,WIDTH)
        policy_history : 각 정책 반복 후 정책 스냅샷      list of (HEIGHT,WIDTH)
        sweeps_per_iter: E-step 당 스윕 수                list of int
        n_changes_per_iter: 각 I-step에서 바뀐 행동 수    list of int
    """
    # ── 초기화: 균등 랜덤으로 결정적 정책 생성 ─────────
    rng = np.random.default_rng(42)
    policy = rng.integers(0, N_ACTIONS, size=N_STATES)
    # 종료 상태는 행동=0으로 고정 (어차피 사용 안 됨)
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            policy[s] = 0

    V = np.zeros(N_STATES)

    V_history       = []
    policy_history  = []
    sweeps_per_iter = []
    n_changes_hist  = []

    for outer in range(max_outer):
        # ──────── E-step: 정책 평가 ────────────────────
        V, n_sweeps = policy_evaluation_step(env, V, policy, gamma, theta)
        V_history.append(V.reshape(HEIGHT, WIDTH).copy())

        # ──────── I-step: 정책 개선 ────────────────────
        policy_new = policy_improvement_step(env, V, gamma)

        # 바뀐 행동 수 계산
        n_changes = int(np.sum(policy != policy_new))

        # 정책 저장 (개선 후)
        policy_history.append(policy_new.reshape(HEIGHT, WIDTH).copy())
        sweeps_per_iter.append(n_sweeps)
        n_changes_hist.append(n_changes)

        print(f"  정책 반복 {outer+1:2d}:  E-step {n_sweeps:4d} sweeps  |  "
              f"I-step {n_changes:2d}개 행동 변경")

        if n_changes == 0:
            print(f"\n  ✓ 수렴 완료: {outer+1}번 정책 반복  "
                  f"(총 {sum(sweeps_per_iter)}번 E-step sweep)")
            break

        policy = policy_new

    return V_history, policy_history, sweeps_per_iter, n_changes_hist


# ══════════════════════════════════════════════════
# 3. 출력 함수
# ══════════════════════════════════════════════════
def print_policy_grid(policy_grid, title):
    print(f"\n  {title}")
    print("  ┌──────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell in ('H', 'G'):
                row_str += f"  {cell} "
            else:
                row_str += f"  {ACTION_NAMES[policy_grid[r,c]]} "
        print(row_str + " │")
    print("  └──────────────────┘")


# ══════════════════════════════════════════════════
# 4. 시각화
# ══════════════════════════════════════════════════
def draw_value_policy(ax, V_grid, policy_grid, title, cmap_name='RdYlGn'):
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
                ax.text(c, r, "H", ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#1565C0', zorder=2))
                ax.text(c, r, "G", ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=3)
            else:
                ax.text(c, r - 0.18, f"{V_grid[r,c]:.3f}",
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', color='black', zorder=3)
                a = policy_grid[r, c]
                ax.text(c, r + 0.22, ACTION_NAMES[a],
                        ha='center', va='center',
                        fontsize=14, color='#1565C0', fontweight='bold', zorder=4)

    ax.set_xticks(range(WIDTH));  ax.set_xticklabels([f"c{c}" for c in range(WIDTH)], fontsize=8)
    ax.set_yticks(range(HEIGHT)); ax.set_yticklabels([f"r{r}" for r in range(HEIGHT)], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    return im


def visualize(V_history, policy_history, sweeps_per_iter, n_changes_hist):
    n_iters = len(V_history)

    # Show at most 5 snapshots
    show_indices = list(range(min(n_iters, 5)))
    if n_iters > 5 and (n_iters - 1) not in show_indices:
        show_indices[-1] = n_iters - 1   # always show last iteration

    n_show = len(show_indices)

    fig = plt.figure(figsize=(4 * n_show + 2, 16))
    fig.suptitle(
        f"Policy Iteration  —  E-step (Policy Evaluation) + I-step (Policy Improvement)\n"
        f"is_slippery=False  |  gamma={GAMMA}  |  "
        f"{n_iters} policy iterations,  {sum(sweeps_per_iter)} E-step sweeps total",
        fontsize=13, fontweight='bold'
    )

    # GridSpec: top row (snapshots) + two bottom rows (charts)
    gs = gridspec.GridSpec(3, n_show, figure=fig,
                           hspace=0.55, wspace=0.35,
                           height_ratios=[1, 1.2, 1])

    # ── Row 0: Value function snapshots per iteration ─
    for col_idx, i in enumerate(show_indices):
        ax = fig.add_subplot(gs[0, col_idx])
        is_last = (i == n_iters - 1)
        label = "pi*" if is_last else f"pi_{i}"
        draw_value_policy(ax, V_history[i], policy_history[i],
            f"Iter {i+1}  V_{label}(s)\n"
            f"E:{sweeps_per_iter[i]}sw  I:{n_changes_hist[i]}ch",
            cmap_name='RdYlGn')

    # ── Row 1: E-step sweeps + I-step action changes ──
    ax_sw = fig.add_subplot(gs[1, :n_show // 2 + 1])
    iters = list(range(1, n_iters + 1))
    bars  = ax_sw.bar(iters, sweeps_per_iter, color='#1976D2', alpha=0.8,
                      label='E-step sweeps per iteration')
    ax_sw.set_xlabel("Policy Iteration Number", fontsize=11)
    ax_sw.set_ylabel("E-step Sweep Count", fontsize=11, color='#1976D2')
    ax_sw.set_title("(A) E-step Sweeps per Policy Iteration\n"
                    "(Sweeps needed to evaluate current policy)",
                    fontsize=11, fontweight='bold')
    ax_sw.tick_params(axis='y', labelcolor='#1976D2')
    for bar, val in zip(bars, sweeps_per_iter):
        ax_sw.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_sw.legend(fontsize=9)
    ax_sw.grid(True, axis='y', alpha=0.3)

    ax_ch = fig.add_subplot(gs[1, n_show // 2 + 1:])
    colors = ['#E53935' if c > 0 else '#2E7D32' for c in n_changes_hist]
    bars2  = ax_ch.bar(iters, n_changes_hist, color=colors, alpha=0.85)
    ax_ch.set_xlabel("Policy Iteration Number", fontsize=11)
    ax_ch.set_ylabel("Actions Changed", fontsize=11, color='#E53935')
    ax_ch.set_title("(B) Action Changes per I-step\n"
                    "(0 = converged = optimal policy pi* reached)",
                    fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, n_changes_hist):
        color = '#2E7D32' if val == 0 else '#B71C1C'
        ax_ch.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                   str(val), ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color=color)
    ax_ch.grid(True, axis='y', alpha=0.3)
    legend_elems = [
        mpatches.Patch(color='#E53935', label='Policy changed'),
        mpatches.Patch(color='#2E7D32', label='Converged (no change)'),
    ]
    ax_ch.legend(handles=legend_elems, fontsize=9)

    # ── Row 2: Cumulative E-step sweep count ──────────
    ax_cum = fig.add_subplot(gs[2, :])
    cum_sweeps = np.cumsum(sweeps_per_iter)
    ax_cum.plot(iters, cum_sweeps, 'o-', color='#6A1B9A', linewidth=2,
                markersize=8, label='Cumulative E-step sweeps')
    for x, y in zip(iters, cum_sweeps):
        ax_cum.annotate(str(y), (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')
    ax_cum.set_xlabel("Policy Iteration Number", fontsize=11)
    ax_cum.set_ylabel("Cumulative E-step Sweeps", fontsize=11)
    ax_cum.set_title("(C) Cumulative E-step Sweeps  —  Total Computation Cost\n"
                     "(Drawback of PI: heavy E-step cost in early iterations)",
                     fontsize=11, fontweight='bold')
    ax_cum.legend(fontsize=10)
    ax_cum.grid(True, alpha=0.3)

    plt.savefig("./c_policy_iteration_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: c_policy_iteration_img.png")


def main():
    print("\n" + "=" * 60)
    print("  정책 반복 (Policy Iteration)")
    print("  E-step (정책 평가) + I-step (정책 개선) 반복")
    print("=" * 60)

    V_history, policy_history, sweeps_per_iter, n_changes_hist = \
        policy_iteration(env, gamma=GAMMA, theta=1e-6)

    # ── 최종 결과 출력 ────────────────────────────────
    print(f"\n  총 정책 반복 횟수  : {len(V_history)}")
    print(f"  총 E-step sweep 수: {sum(sweeps_per_iter)}")

    print_policy_grid(policy_history[-1], "최적 정책 π*")

    print(f"\n  최적 가치 함수 V*(s)  (γ={GAMMA})")
    print("       col0     col1     col2     col3")
    V_final = V_history[-1]
    for r in range(HEIGHT):
        row_str = f"  row{r}"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            row_str += f"  {'['+cell+']':^7}" if cell in ('H', 'G') \
                       else f"  {V_final[r,c]:7.4f}"
        print(row_str)

    # ── 시각화 ────────────────────────────────────────
    visualize(V_history, policy_history, sweeps_per_iter, n_changes_hist)

    env.close()


if __name__ == "__main__":
    main()
