"""
Frozen Lake (is_slippery=False) — Optimal State Value Function
Value Iteration using Bellman Optimality Equation

Bellman Optimality Equation:
  v*(s) = max_a  Σ_{s'} p(s'|s,a) * [r + γ * v*(s')]

Comparison:
  Bellman Expectation  →  v_π(s) = Σ_a π(a|s) Σ_{s'} p(s'|s,a)[r + γv_π(s')]
  Bellman Optimality   →  v*(s)  = max_a       Σ_{s'} p(s'|s,a)[r + γv*(s')]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

from a_state_values import bellman_expectation

# ══════════════════════════════════════════════════
# 1. Environment Setup
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]

GAMMA     = 0.99
HEIGHT    = 4
WIDTH     = 4
N_STATES  = HEIGHT * WIDTH
N_ACTIONS = 4   # LEFT=0, DOWN=1, RIGHT=2, UP=3
PI        = 0.25  # uniform random policy (for comparison)

ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# 셀 타입 맵
CELL = {}
for r in range(HEIGHT):
    for c in range(WIDTH):
        CELL[(r, c)] = MAP_4x4[r][c]

env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)


# ══════════════════════════════════════════════════
# 1. Bellman Optimality
#    v*(s) = max_a  Σ_{s'} p(s'|s,a)[r + γ v*(s')]
# ══════════════════════════════════════════════════
def bellman_optimality_one_step(env, V, gamma=GAMMA):
    """
    Bellman Optimality Equation — one sweep over all states

    Returns:
        V_new (np.ndarray): 갱신된 가치 함수 (flat, N_STATES)
        delta (float):      이번 스윕의 최대 변화량
    """
    delta = 0.0
    V_new = np.copy(V)

    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            continue

        # Q(s,a) for each action → take max
        q_values = []
        for a in range(N_ACTIONS):
            q_sa = 0.0
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                q_sa += prob * (reward + gamma * V[ns])
            q_values.append(q_sa)

        v_new = max(q_values)           # ← Bellman Optimality

        delta    = max(delta, abs(V[s] - v_new))
        V_new[s] = v_new

    return V_new, delta


def bellman_optimality(env, gamma=GAMMA, theta=1e-9, max_iter=100000):
    V = np.zeros(N_STATES)
    delta_hist = []

    for iteration in range(max_iter):
        V, delta = bellman_optimality_one_step(env, V, gamma)
        delta_hist.append(delta)
        if delta < theta:
            print(f"  [Bellman Optimality]  Converged at iter {iteration+1}"
                  f"  (Δ={delta:.2e})")
            break

    return V.reshape(HEIGHT, WIDTH), delta_hist


# ══════════════════════════════════════════════════
# 2. Extract Optimal Policy from v*
#    π*(s) = argmax_a  Σ_{s'} p(s'|s,a)[r + γ v*(s')]
# ══════════════════════════════════════════════════
def extract_policy(env, V_star, gamma=GAMMA):
    policy = np.full((HEIGHT, WIDTH), -1, dtype=int)

    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            continue

        q_values = []
        for a in range(N_ACTIONS):
            q_sa = 0.0
            for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                q_sa += prob * (reward + gamma * V_star.flatten()[ns])
            q_values.append(q_sa)

        policy[r, c] = int(np.argmax(q_values))

    return policy


def print_value_function(V_exp, V_opt, policy_opt):
    print(f"\n  Bellman Expectation  V_π(s)  [Uniform Random π=0.25, γ={GAMMA}]")
    print("         col0     col1     col2     col3")
    for r in range(HEIGHT):
        row_str = f"  row{r}"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            row_str += f"  {'['+cell+']':^7}" if cell in ('H','G') \
                       else f"  {V_exp[r,c]:7.4f}"
        print(row_str)

    print(f"\n  Bellman Optimality   V*(s)   [Optimal policy, γ={GAMMA}]")
    print("         col0     col1     col2     col3")
    for r in range(HEIGHT):
        row_str = f"  row{r}"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            row_str += f"  {'['+cell+']':^7}" if cell in ('H','G') \
                       else f"  {V_opt[r,c]:7.4f}"
        print(row_str)

    print("\n  Optimal Policy  π*(s)  [action arrows]")
    print("  ┌──────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                row_str += "  H "
            elif cell == 'G':
                row_str += "  G "
            else:
                row_str += f"  {ACTION_NAMES[policy_opt[r,c]]} "
        print(row_str + " │")
    print("  └──────────────────┘")


# ── helper: draw value heatmap ───────────────────
def draw_value_heatmap(ax, V_grid, title, show_policy=False, policy=None):
    V_plot = V_grid.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                V_plot[r, c] = np.nan

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='#bdbdbd')
    vmax = np.nanmax(np.abs(V_plot)) + 1e-9
    im = ax.imshow(V_plot, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                             color='#37474F', zorder=2))
                ax.text(c, r, "H\n(Hole)", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                             color='#1565C0', zorder=2))
                ax.text(c, r, "G\n(Goal)", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            else:
                label = "S\n" if cell == 'S' else ""
                ax.text(c, r - 0.15 if label else r,
                        f"{label}{V_grid[r,c]:.4f}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='black', zorder=3)
                # draw optimal action arrow
                if show_policy and policy is not None:
                    a = policy[r, c]
                    ax.text(c, r + 0.32, ACTION_NAMES[a],
                            ha='center', va='center', fontsize=12,
                            color='#1565C0', fontweight='bold', zorder=4)

    ax.set_xticks(range(WIDTH))
    ax.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax.set_yticks(range(HEIGHT))
    ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.82, label="Value")


def visualize(env, V_exp, V_opt, delta_exp, delta_opt, policy_opt):
    # ══════════════════════════════════════════════════
    # 5. Visualization
    # ══════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Frozen Lake 4×4  (is_slippery=False, γ={GAMMA})\n"
        "Bellman Expectation Equation  vs.  Bellman Optimality Equation",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── (A) Bellman Expectation  V_π ─────────────────
    ax_exp = fig.add_subplot(gs[0, 0])
    draw_value_heatmap(ax_exp, V_exp,
        f"(A) Bellman Expectation  V_π(s)\n"
        f"Uniform Random Policy  π=0.25  |  γ={GAMMA}")

    # ── (B) Bellman Optimality  V* ───────────────────
    ax_opt = fig.add_subplot(gs[0, 1])
    draw_value_heatmap(ax_opt, V_opt,
        f"(B) Bellman Optimality  V*(s)\n"
        f"Optimal Policy  |  γ={GAMMA}",
        show_policy=True, policy=policy_opt)

    # ── (C) Difference  V* - V_π ─────────────────────
    ax_diff = fig.add_subplot(gs[0, 2])
    diff = V_opt - V_exp
    diff_plot = diff.copy().astype(float)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] in ('H', 'G'):
                diff_plot[r, c] = np.nan

    cmap_diff = plt.cm.Blues.copy()
    cmap_diff.set_bad(color='#bdbdbd')
    im_d = ax_diff.imshow(diff_plot, cmap=cmap_diff, aspect='auto')
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax_diff.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                  color='#37474F', zorder=2))
                ax_diff.text(c, r, "H", ha='center', va='center',
                             fontsize=10, color='white', zorder=3)
            elif cell == 'G':
                ax_diff.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                  color='#1565C0', zorder=2))
                ax_diff.text(c, r, "G", ha='center', va='center',
                             fontsize=10, color='white', zorder=3)
            else:
                ax_diff.text(c, r, f"+{diff[r,c]:.4f}" if diff[r,c] >= 0
                             else f"{diff[r,c]:.4f}",
                             ha='center', va='center', fontsize=10,
                             fontweight='bold', color='black', zorder=3)
    ax_diff.set_xticks(range(WIDTH))
    ax_diff.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_diff.set_yticks(range(HEIGHT))
    ax_diff.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_diff.set_title("(C) Improvement  V*(s) - V_π(s)\n"
                      "(How much better is optimal?)",
                      fontsize=11, fontweight='bold')
    plt.colorbar(im_d, ax=ax_diff, shrink=0.82, label="V* - V_π")

    # ── (D) Convergence Curves ───────────────────────
    ax_conv = fig.add_subplot(gs[1, :2])
    ax_conv.semilogy(delta_exp, color='#1976D2', linewidth=2,
                     label=f'Bellman Expectation  ({len(delta_exp)} iters)')
    ax_conv.semilogy(delta_opt, color='#E53935', linewidth=2,
                     label=f'Bellman Optimality   ({len(delta_opt)} iters)')
    ax_conv.axhline(1e-9, color='gray', linestyle='--', linewidth=1,
                    label='Convergence threshold  θ=1e-9')
    ax_conv.set_xlabel("Iteration", fontsize=11)
    ax_conv.set_ylabel("Max Delta  Δ  (log scale)", fontsize=11)
    ax_conv.set_title("(D) Convergence Comparison\n"
                      "Bellman Expectation vs. Bellman Optimality",
                      fontsize=11, fontweight='bold')
    ax_conv.legend(fontsize=10)
    ax_conv.grid(True, which='both', alpha=0.3)

    # ── (E) Optimal Policy Grid ──────────────────────
    ax_pol = fig.add_subplot(gs[1, 2])

    colors_cell = {'S': '#E8F5E9', 'F': '#E8F5E9',
                   'H': '#37474F', 'G': '#1565C0'}

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            ax_pol.add_patch(plt.Rectangle(
                (c-.5, r-.5), 1, 1,
                facecolor=colors_cell[cell], zorder=1,
                edgecolor='white', linewidth=2))

            if cell == 'H':
                ax_pol.text(c, r, "H\n(Hole)", ha='center', va='center',
                            fontsize=11, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax_pol.text(c, r, "G\n(Goal)", ha='center', va='center',
                            fontsize=11, color='white', fontweight='bold', zorder=3)
            else:
                a = policy_opt[r, c]
                ax_pol.text(c, r, ACTION_NAMES[a], ha='center', va='center',
                            fontsize=22, color='#1565C0',
                            fontweight='bold', zorder=3)
                ax_pol.text(c, r + 0.35, ACTION_NAMES[a],
                            ha='center', va='center', fontsize=8,
                            color='#555', zorder=3)

    ax_pol.set_xlim(-0.5, WIDTH - 0.5)
    ax_pol.set_ylim(HEIGHT - 0.5, -0.5)
    ax_pol.set_xticks(range(WIDTH))
    ax_pol.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax_pol.set_yticks(range(HEIGHT))
    ax_pol.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax_pol.set_title("(E) Optimal Policy  π*(s)\n"
                     "Greedy w.r.t. V*(s)",
                     fontsize=11, fontweight='bold')

    legend_elems = [
        mpatches.Patch(color='#37474F', label='H: Hole  (terminal, fail)'),
        mpatches.Patch(color='#1565C0', label='G: Goal  (terminal, success)'),
        mpatches.Patch(color='#E8F5E9', label='S/F: Safe  (arrows = π*)'),
    ]
    ax_pol.legend(handles=legend_elems, loc='lower right',
                  fontsize=7.5, framealpha=0.9)

    plt.savefig("./c_optimal_state_values_img.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: c_optimal_state_values_img.png")


def main():
    # ── Run both algorithms ──────────────────────────
    V_exp, delta_exp = bellman_expectation(env, gamma=GAMMA)   # Expectation
    V_opt, delta_opt = bellman_optimality(env, gamma=GAMMA)    # Optimality
    policy_opt = extract_policy(env, V_opt)

    # ── Console output ───────────────────────────────
    print_value_function(V_exp, V_opt, policy_opt)

    # ── Visualization ────────────────────────────────
    visualize(env, V_exp, V_opt, delta_exp, delta_opt, policy_opt)

    env.close()


if __name__ == "__main__":
    main()
