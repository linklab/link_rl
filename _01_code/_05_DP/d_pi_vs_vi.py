"""
Frozen Lake — 정책 반복 vs. 가치 반복 + 일반화된 정책 반복 (GPI)
Policy Iteration vs. Value Iteration + Generalized Policy Iteration

05.DP.pptx 슬라이드 16-22 대응

핵심 개념:

  [가치 반복 (Value Iteration) — 슬라이드 16-18]
    v_{k+1}(s) = max_a  Σ_{s'} p(s'|s,a)[r + γ·v_k(s')]
    - 정책 평가를 "한 번만" 수행 (E-step = 1 sweep)
    - 벨만 최적 방정식을 직접 반복 적용
    - 가치가 충분히 수렴하면 greedy 정책 추출

  [정책 반복 vs. 가치 반복 비교 — 슬라이드 19]
    정책 반복: 완전한 E-step (수렴까지) + I-step 반복
    가치 반복: 1번 E-step(1 sweep, 최적 방정식 사용) 반복

  [일반화된 정책 반복 GPI — 슬라이드 20-22]
    - E-step과 I-step의 크기를 일반화
    - 정책 반복과 가치 반복은 GPI의 양 극단
    - 대부분의 RL 방법론은 GPI 프레임워크로 설명 가능

  비교 시각화:
    (A) 정책 반복: 최종 가치 함수 + 최적 정책
    (B) 가치 반복: 최종 가치 함수 + 최적 정책
    (C) 두 방법의 가치 함수 차이
    (D) 누적 sweep 수 비교 (수렴 속도)
    (E) GPI 프레임워크 개념 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch
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
N_ACTIONS = 4

ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}

CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)


# ══════════════════════════════════════════════════
# 2. 공통 유틸리티
# ══════════════════════════════════════════════════
def extract_greedy_policy(env, V_flat, gamma=GAMMA):
    """V_flat → 탐욕 결정적 정책  (N_STATES,)"""
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            continue
        q_vals = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                q_vals[a] += prob * (reward + gamma * V_flat[ns])
        policy[s] = int(np.argmax(q_vals))
    return policy


# ══════════════════════════════════════════════════
# 3. 정책 반복 (Policy Iteration)
#    E-step: 완전 수렴까지  (theta=1e-6)
#    I-step: greedy 개선
# ══════════════════════════════════════════════════
def policy_iteration(env, gamma=GAMMA, theta_eval=1e-6, max_outer=100):
    """
    Returns:
        V_star  : shape (HEIGHT, WIDTH)  최적 가치 함수
        policy  : shape (HEIGHT, WIDTH)  최적 정책
        total_sweeps_hist : 각 반복 후 누적 sweep 수
        V_snapshots : 각 정책 반복 후 V 스냅샷 (for GPI 시각화)
    """
    rng = np.random.default_rng(0)
    policy = rng.integers(0, N_ACTIONS, size=N_STATES)
    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            policy[s] = 0

    V = np.zeros(N_STATES)
    total_sweeps = 0
    total_sweeps_hist = []
    V_snapshots = []

    for outer in range(max_outer):
        # ── E-step: 완전 정책 평가 ───────────────────
        sweeps = 0
        while True:
            delta = 0.0
            V_new = np.copy(V)
            for s in range(N_STATES):
                r, c = divmod(s, WIDTH)
                if CELL[(r, c)] in ('H', 'G'):
                    continue
                a = policy[s]
                v_new = 0.0
                for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                    v_new += prob * (reward + gamma * V[ns])
                delta    = max(delta, abs(V[s] - v_new))
                V_new[s] = v_new
            V = V_new
            sweeps += 1
            if delta < theta_eval:
                break
        total_sweeps += sweeps
        V_snapshots.append(('E', V.reshape(HEIGHT, WIDTH).copy()))

        # ── I-step: 탐욕 정책 개선 ───────────────────
        policy_new = extract_greedy_policy(env, V, gamma)
        n_changed  = int(np.sum(policy != policy_new))
        V_snapshots.append(('I', V.reshape(HEIGHT, WIDTH).copy()))

        total_sweeps_hist.append(total_sweeps)
        policy = policy_new

        if n_changed == 0:
            print(f"  [PI] 수렴: {outer+1}번 외부 반복, 총 {total_sweeps} sweeps")
            break

    return (V.reshape(HEIGHT, WIDTH), policy.reshape(HEIGHT, WIDTH),
            total_sweeps_hist, V_snapshots)


# ══════════════════════════════════════════════════
# 4. 가치 반복 (Value Iteration)
#    벨만 최적 방정식 한 번씩만 적용 (1 sweep = 1 iter)
#    v_{k+1}(s) = max_a Σ_{s'} p(s'|s,a)[r + γv_k(s')]
# ══════════════════════════════════════════════════
def value_iteration(env, gamma=GAMMA, theta=1e-9, max_iter=100000):
    """
    Returns:
        V_star  : shape (HEIGHT, WIDTH)
        policy  : shape (HEIGHT, WIDTH)
        delta_history : 각 sweep의 최대 변화량
        V_snapshots   : 일부 V 스냅샷
    """
    V = np.zeros(N_STATES)
    delta_history = []
    V_snapshots   = []
    snap_iters    = set()  # 스냅샷 저장 시점

    for iteration in range(max_iter):
        delta = 0.0
        V_new = np.copy(V)

        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            if CELL[(r, c)] in ('H', 'G'):
                continue
            # 벨만 최적 방정식: max_a Q(s,a)
            q_vals = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                    q_vals[a] += prob * (reward + gamma * V[ns])
            v_new = np.max(q_vals)       # ← 정책 반복의 Σ_a π(a|s)·... 과의 차이점
            delta    = max(delta, abs(V[s] - v_new))
            V_new[s] = v_new

        V = V_new
        delta_history.append(delta)

        if delta < theta:
            print(f"  [VI] 수렴: {iteration+1} sweeps  (Δ={delta:.2e})")
            break

    policy = extract_greedy_policy(env, V, gamma)
    return (V.reshape(HEIGHT, WIDTH), policy.reshape(HEIGHT, WIDTH),
            delta_history)


# ══════════════════════════════════════════════════
# 5. 출력 함수
# ══════════════════════════════════════════════════
def print_comparison(V_pi, policy_pi, V_vi, policy_vi):
    print(f"\n{'='*72}")
    print(f"  정책 반복 (PI) vs. 가치 반복 (VI)  최종 결과 비교  (γ={GAMMA})")
    print(f"{'='*72}")
    print(f"  {'상태':^8}  {'V_PI':^10}  {'V_VI':^10}  "
          f"{'π_PI':^6}  {'π_VI':^6}  {'동일?':^6}")
    print("  " + "-" * 56)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            s_label = f"({r},{c})[{cell}]"
            if cell in ('H', 'G'):
                print(f"  {s_label:^8}  {'terminal':^10}  {'terminal':^10}")
                continue
            vp = V_pi[r, c]
            vv = V_vi[r, c]
            ap = ACTION_NAMES[policy_pi[r, c]]
            av = ACTION_NAMES[policy_vi[r, c]]
            same = "✓" if policy_pi[r,c] == policy_vi[r,c] else "✗"
            print(f"  {s_label:^8}  {vp:^10.6f}  {vv:^10.6f}  "
                  f"{ap:^6}  {av:^6}  {same:^6}")
    print(f"\n  최대 V 차이: {np.max(np.abs(V_pi - V_vi)):.2e}  "
          f"(두 알고리즘의 결과는 수치적으로 동일)")


# ══════════════════════════════════════════════════
# 6. 시각화
# ══════════════════════════════════════════════════
def draw_vp_map(ax, V_grid, policy_grid, title, cmap_name='RdYlGn'):
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
                ax.text(c, r - 0.18, f"{V_grid[r,c]:.4f}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='black', zorder=3)
                a = policy_grid[r, c]
                ax.text(c, r + 0.22, ACTION_NAMES[a],
                        ha='center', va='center',
                        fontsize=14, color='#1565C0', fontweight='bold', zorder=4)

    ax.set_xticks(range(WIDTH));  ax.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax.set_yticks(range(HEIGHT)); ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.82, label="V*(s)")


def draw_gpi_diagram(ax):
    """
    Generalized Policy Iteration (GPI) diagram
    Interplay between E-step and I-step
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("(E) Generalized Policy Iteration (GPI)\n"
                 "E-step (Eval) <-> I-step (Improve) interplay",
                 fontsize=11, fontweight='bold')

    # ── Optimal convergence point ────────────────────
    ax.plot(5, 5, 'o', color='gold', markersize=20, zorder=5, markeredgecolor='#333')
    ax.text(5, 5, "pi*\nV*", ha='center', va='center',
            fontsize=10, fontweight='bold', zorder=6)
    ax.text(5, 3.8, "Optimal point", ha='center', va='center',
            fontsize=9, color='#555')

    # ── V axis (vertical) ────────────────────────────
    ax.annotate('', xy=(5, 9.5), xytext=(5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
    ax.text(5.3, 9.2, "V (Value fn)", fontsize=9, color='#1976D2', rotation=90)

    # ── pi axis (horizontal) ─────────────────────────
    ax.annotate('', xy=(9.5, 5), xytext=(0.5, 5),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
    ax.text(8.8, 5.3, "pi (Policy)", fontsize=9, color='#E53935')

    # ── Convergence trajectory ────────────────────────
    traj = [
        (1.5, 1.5),   # initial
        (1.5, 3.5),   # E-step 1 (V update)
        (3.0, 3.5),   # I-step 1 (pi update)
        (3.0, 4.5),   # E-step 2
        (4.0, 4.5),   # I-step 2
        (4.0, 4.9),   # E-step 3
        (4.7, 4.9),   # I-step 3
        (5.0, 5.0),   # converged
    ]
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]

    # E-step segments: vertical moves
    e_segs = [(0,1), (2,3), (4,5), (6,7)]
    i_segs = [(1,2), (3,4), (5,6)]

    for seg in e_segs:
        i0, i1 = seg
        ax.annotate('', xy=(xs[i1], ys[i1]), xytext=(xs[i0], ys[i0]),
                    arrowprops=dict(arrowstyle='->', color='#1976D2',
                                    lw=2.5, shrinkA=0, shrinkB=0))
    for seg in i_segs:
        i0, i1 = seg
        ax.annotate('', xy=(xs[i1], ys[i1]), xytext=(xs[i0], ys[i0]),
                    arrowprops=dict(arrowstyle='->', color='#E53935',
                                    lw=2.5, shrinkA=0, shrinkB=0))

    # Initial point
    ax.plot(xs[0], ys[0], 's', color='#795548', markersize=10, zorder=5)
    ax.text(xs[0]-0.1, ys[0]-0.5, "Initial\npi0, V0", ha='center', va='center',
            fontsize=8, color='#795548')

    # Legend
    legend_elems = [
        mpatches.Patch(color='#1976D2', label='E-step: Policy Eval (V dir)'),
        mpatches.Patch(color='#E53935', label='I-step: Policy Impr (pi dir)'),
        mpatches.Patch(color='gold',    label='Convergence: optimal (V*, pi*)'),
    ]
    ax.legend(handles=legend_elems, loc='upper left',
              fontsize=8.5, framealpha=0.9)

    # GPI definition text
    ax.text(0.5, 0.2,
            "GPI: Generalize E-step/I-step size & frequency\n"
            "· Policy Iteration: E-step=full convergence\n"
            "· Value Iteration:  E-step=1 sweep only\n"
            "Most RL algorithms fit the GPI framework",
            fontsize=8.5, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4',
                      edgecolor='#F9A825', linewidth=1.2))


def visualize(V_pi, policy_pi, total_sweeps_pi,
              V_vi, policy_vi, delta_vi):
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Policy Iteration (PI) vs. Value Iteration (VI)  +  Generalized Policy Iteration (GPI)\n"
        f"is_slippery=False  |  gamma={GAMMA}",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── (A) Policy Iteration final result ────────────
    ax_a = fig.add_subplot(gs[0, 0])
    draw_vp_map(ax_a, V_pi, policy_pi,
        f"(A) Policy Iteration (PI)\n"
        f"Final V*(s) + pi*(s)  |  {total_sweeps_pi[-1]} sweeps total",
        cmap_name='RdYlGn')

    # ── (B) Value Iteration final result ─────────────
    ax_b = fig.add_subplot(gs[0, 1])
    draw_vp_map(ax_b, V_vi, policy_vi,
        f"(B) Value Iteration (VI)\n"
        f"Final V*(s) + pi*(s)  |  {len(delta_vi)} sweeps total",
        cmap_name='YlGn')

    # ── (C) Difference between two methods ───────────
    ax_c = fig.add_subplot(gs[0, 2])
    diff = np.abs(V_pi - V_vi)
    diff_plot = diff.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                diff_plot[r, c] = np.nan

    cmap_d = plt.cm.Purples.copy()
    cmap_d.set_bad(color='#cccccc')
    im_c = ax_c.imshow(diff_plot, cmap=cmap_d,
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
                ax_c.text(c, r, f"{diff[r,c]:.2e}",
                          ha='center', va='center', fontsize=9,
                          fontweight='bold', color='black', zorder=3)
    ax_c.set_xticks(range(WIDTH));  ax_c.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_c.set_yticks(range(HEIGHT)); ax_c.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_c.set_title("(C) |V_PI - V_VI|\n(Numerical difference between the two algorithms)",
                   fontsize=11, fontweight='bold')
    plt.colorbar(im_c, ax=ax_c, shrink=0.82, label="|V_PI - V_VI|")

    # ── (D) Convergence speed comparison ─────────────
    ax_d = fig.add_subplot(gs[1, :2])

    # PI: cumulative sweep count (step-wise)
    pi_x = list(range(1, len(total_sweeps_pi) + 1))
    pi_y = total_sweeps_pi
    ax_d.step(pi_x, pi_y, where='post', color='#1976D2', linewidth=2.5,
              label=f'Policy Iteration (PI): {total_sweeps_pi[-1]} sweeps total', linestyle='-')
    ax_d.plot(pi_x, pi_y, 'o', color='#1976D2', markersize=8)

    # VI: cumulative sweep = iteration count (1:1)
    vi_x = list(range(1, len(delta_vi) + 1))
    vi_y = list(range(1, len(delta_vi) + 1))
    ax_d.plot(vi_x, vi_y, color='#E53935', linewidth=2.5,
              label=f'Value Iteration (VI): {len(delta_vi)} sweeps total', linestyle='--')

    ax_d.set_xlabel("PI Policy Iter / VI Iteration Number", fontsize=11)
    ax_d.set_ylabel("Cumulative E-step Sweeps", fontsize=11)
    ax_d.set_title("(D) Cumulative Sweep Count Comparison\n"
                   "PI: multiple E-step sweeps per iter  |  "
                   "VI: 1 sweep per iter (faster)", fontsize=11, fontweight='bold')
    ax_d.legend(fontsize=11)
    ax_d.grid(True, alpha=0.3)

    # VI convergence curve (right axis)
    ax_d2 = ax_d.twinx()
    ax_d2.semilogy(vi_x, delta_vi, color='#FF8F00', linewidth=1.5,
                   linestyle=':', label='VI Delta (log)')
    ax_d2.axhline(1e-9, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_d2.set_ylabel("VI Max Delta (log scale)", fontsize=9, color='#FF8F00')
    ax_d2.tick_params(axis='y', labelcolor='#FF8F00')
    ax_d2.legend(fontsize=9, loc='upper right')

    # ── (E) GPI 개념도 ────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    draw_gpi_diagram(ax_e)

    plt.savefig("./d_pi_vs_vi_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: d_pi_vs_vi_img.png")


def main():
    print("\n" + "=" * 65)
    print("  정책 반복 (PI) vs. 가치 반복 (VI)  +  GPI")
    print("=" * 65)

    # ── 정책 반복 실행 ────────────────────────────────
    print("\n[1] 정책 반복 (Policy Iteration)")
    V_pi, policy_pi, sweeps_hist_pi, _ = policy_iteration(env)

    # ── 가치 반복 실행 ────────────────────────────────
    print("\n[2] 가치 반복 (Value Iteration)")
    V_vi, policy_vi, delta_vi = value_iteration(env)

    # ── 비교 출력 ─────────────────────────────────────
    print_comparison(V_pi, policy_pi, V_vi, policy_vi)

    print(f"\n  계산 비용 비교:")
    print(f"    정책 반복 (PI): 총 {sweeps_hist_pi[-1]} E-step sweeps  "
          f"({len(sweeps_hist_pi)}번 정책 반복)")
    print(f"    가치 반복 (VI): 총 {len(delta_vi)} sweeps  "
          f"(1 sweep/iter, theta=1e-9)")

    print(f"\n  GPI 관점:")
    print(f"    정책 반복 = E-step을 완전히 수렴시키는 GPI")
    print(f"    가치 반복 = E-step을 1 sweep만 하는 GPI")
    print(f"    두 방법 모두 같은 최적 해에 수렴함")

    # ── 시각화 ────────────────────────────────────────
    visualize(V_pi, policy_pi, sweeps_hist_pi,
              V_vi, policy_vi, delta_vi)

    env.close()


if __name__ == "__main__":
    main()
