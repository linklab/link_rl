"""
Frozen Lake (is_slippery=False) — Optimal State-Action Value Function (Q*)
Q-Value Iteration using Bellman Optimality Equation for Q

Bellman Optimality Equation for Q:
  q*(s,a) = Σ_{s'} p(s'|s,a) * [r + γ * max_{a'} q*(s',a')]

Comparison:
  Bellman Expectation  →  q_π(s,a) = Σ_{s'} p(s'|s,a)[r + γ Σ_{a'} π(a'|s') q_π(s',a')]
  Bellman Optimality   →  q*(s,a)  = Σ_{s'} p(s'|s,a)[r + γ max_{a'}       q*(s', a')]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import platform
import gymnasium as gym

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════
# 1. Environment Setup
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]

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
# 2. Iterative Policy Evaluation  (Bellman Expectation for Q)
#    q_π(s,a) = Σ_{s'} p(s'|s,a)[r + γ Σ_{a'} π(a'|s') q_π(s',a')]
# ══════════════════════════════════════════════════
def policy_evaluation(env, gamma=GAMMA, theta=1e-9, max_iter=100000):
    Q = np.zeros((N_STATES, N_ACTIONS))
    delta_hist = []

    for iteration in range(max_iter):
        delta = 0.0
        Q_new = np.copy(Q)

        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            if CELL[(r, c)] in ('H', 'G'):
                continue

            for a in range(N_ACTIONS):
                q_new = 0.0
                for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                    # V_π(s') = Σ_{a'} π(a'|s') * Q(s',a')
                    v_next = PI * np.sum(Q[ns, :])
                    q_new += prob * (reward + gamma * v_next)

                delta       = max(delta, abs(Q[s, a] - q_new))
                Q_new[s, a] = q_new

        Q = Q_new
        delta_hist.append(delta)
        if delta < theta:
            print(f"  [Policy Evaluation]  Converged at iter {iteration+1}"
                  f"  (Δ={delta:.2e})")
            break

    return Q.reshape(HEIGHT, WIDTH, N_ACTIONS), delta_hist


# ══════════════════════════════════════════════════
# 3. Q-Value Iteration  (Bellman Optimality for Q)
#    q*(s,a) = Σ_{s'} p(s'|s,a)[r + γ max_{a'} q*(s',a')]
# ══════════════════════════════════════════════════
def value_iteration(env, gamma=GAMMA, theta=1e-9, max_iter=100000):
    Q = np.zeros((N_STATES, N_ACTIONS))
    delta_hist = []

    for iteration in range(max_iter):
        delta = 0.0
        Q_new = np.copy(Q)

        for s in range(N_STATES):
            r, c = divmod(s, WIDTH)
            if CELL[(r, c)] in ('H', 'G'):
                continue

            for a in range(N_ACTIONS):
                q_new = 0.0
                for prob, ns, reward, _ in env.unwrapped.P[s][a]:
                    # max_{a'} q*(s',a')  ← Bellman Optimality
                    q_new += prob * (reward + gamma * np.max(Q[ns, :]))

                delta       = max(delta, abs(Q[s, a] - q_new))
                Q_new[s, a] = q_new

        Q = Q_new
        delta_hist.append(delta)
        if delta < theta:
            print(f"  [Q-Value Iteration]  Converged at iter {iteration+1}"
                  f"  (Δ={delta:.2e})")
            break

    return Q.reshape(HEIGHT, WIDTH, N_ACTIONS), delta_hist


# ══════════════════════════════════════════════════
# 4. Extract Optimal Policy from Q*
#    π*(s) = argmax_a  q*(s,a)
# ══════════════════════════════════════════════════
def extract_policy(env, Q_star, gamma=GAMMA):
    policy = np.full((HEIGHT, WIDTH), -1, dtype=int)

    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        if CELL[(r, c)] in ('H', 'G'):
            continue
        policy[r, c] = int(np.argmax(Q_star[r, c, :]))

    return policy


def print_results(Q_exp, Q_opt, policy_opt):
    ACTION_LABELS = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    for label, Q in [("Bellman Expectation  q_π(s,a)", Q_exp),
                     ("Bellman Optimality   q*(s,a) ", Q_opt)]:
        print(f"\n  {label}  [γ={GAMMA}]")
        print(f"  {'State':<12}  {'LEFT':>8}  {'DOWN':>8}  {'RIGHT':>8}  {'UP':>8}")
        print("  " + "-" * 52)
        for r in range(HEIGHT):
            for c in range(WIDTH):
                cell = CELL[(r, c)]
                s_label = f"({r},{c})[{cell}]"
                if cell in ('H', 'G'):
                    print(f"  {s_label:<12}  {'terminal':>8}")
                else:
                    q = Q[r, c, :]
                    best = int(np.argmax(q))
                    row = f"  {s_label:<12}"
                    for a in range(N_ACTIONS):
                        marker = "*" if a == best else " "
                        row += f"  {q[a]:7.4f}{marker}"
                    print(row)

    print("\n  Optimal Policy  π*(s)  [argmax_a q*(s,a)]")
    print("  ┌──────────────────┐")
    for r in range(HEIGHT):
        row_str = "  │"
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':   row_str += "  H "
            elif cell == 'G': row_str += "  G "
            else:             row_str += f"  {ACTION_NAMES[policy_opt[r,c]]} "
        print(row_str + " │")
    print("  └──────────────────┘")


# ── helper: draw Q heatmap for one action ────────
def draw_q_heatmap(ax, Q_grid, a_idx, title, show_policy=False, policy=None):
    ACTION_ARROWS = ["←", "↓", "→", "↑"]

    Q_a = np.full((HEIGHT, WIDTH), np.nan)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if CELL[(r, c)] not in ('H', 'G'):
                Q_a[r, c] = Q_grid[r, c, a_idx]

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='#bdbdbd')
    vmax = np.nanmax(np.abs(Q_a)) + 1e-9
    im = ax.imshow(Q_a, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                             facecolor='#37474F', zorder=2))
                ax.text(c, r, "H\n(Hole)", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                             facecolor='#1565C0', zorder=2))
                ax.text(c, r, "G\n(Goal)", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=3)
            else:
                is_best = (a_idx == int(np.argmax(Q_grid[r, c, :])))
                color = '#B71C1C' if is_best else 'black'
                ax.text(c, r, f"{Q_grid[r,c,a_idx]:.4f}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color=color, zorder=3)
                if show_policy and policy is not None and is_best:
                    ax.text(c, r + 0.32, ACTION_ARROWS[a_idx],
                            ha='center', va='center', fontsize=12,
                            color='#1565C0', fontweight='bold', zorder=4)

    ax.set_xticks(range(WIDTH))
    ax.set_xticklabels([f"col{c}" for c in range(WIDTH)])
    ax.set_yticks(range(HEIGHT))
    ax.set_yticklabels([f"row{r}" for r in range(HEIGHT)])
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.82, label="Q value")


def visualize(env, Q_exp, Q_opt, delta_exp, delta_opt, policy_opt):
    # ══════════════════════════════════════════════════
    # 5. Visualization
    # ══════════════════════════════════════════════════
    ACTION_ARROWS = ["←", "↓", "→", "↑"]
    ACTION_LABELS = ["L", "D", "R", "U"]

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        f"Frozen Lake 4×4  (is_slippery=False, γ={GAMMA})\n"
        "Bellman Expectation q_π(s,a)  vs.  Bellman Optimality q*(s,a)",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40,
                           height_ratios=[1, 1, 1.6])

    # ── (A) Bellman Expectation q_π — 4 action heatmaps ──
    for a_idx in range(N_ACTIONS):
        ax_a = fig.add_subplot(gs[0, a_idx])
        draw_q_heatmap(ax_a, Q_exp, a_idx,
            f"(A{a_idx+1}) q_π  A={ACTION_ARROWS[a_idx]}{ACTION_LABELS[a_idx]}\n"
            f"Expectation | π=0.25")

    # ── (B) Bellman Optimality q* — 4 action heatmaps ────
    for a_idx in range(N_ACTIONS):
        ax_b = fig.add_subplot(gs[1, a_idx])
        draw_q_heatmap(ax_b, Q_opt, a_idx,
            f"(B{a_idx+1}) q*  A={ACTION_ARROWS[a_idx]}{ACTION_LABELS[a_idx]}\n"
            f"Optimality  (red=argmax)",
            show_policy=True, policy=policy_opt)

    # ── (C) Convergence Curves ────────────────────────────
    ax_conv = fig.add_subplot(gs[2, :2])
    ax_conv.semilogy(delta_exp, color='#1976D2', linewidth=2,
                     label=f'Policy Evaluation  (Expectation, {len(delta_exp)} iters)')
    ax_conv.semilogy(delta_opt, color='#E53935', linewidth=2,
                     label=f'Q-Value Iteration  (Optimality,  {len(delta_opt)} iters)')
    ax_conv.axhline(1e-9, color='gray', linestyle='--', linewidth=1,
                    label='Convergence threshold  θ=1e-9')
    ax_conv.set_xlabel("Iteration", fontsize=11)
    ax_conv.set_ylabel("Max Delta  Δ  (log scale)", fontsize=11)
    ax_conv.set_title("(C) Convergence Comparison\n"
                      "Bellman Expectation vs. Bellman Optimality",
                      fontsize=11, fontweight='bold')
    ax_conv.legend(fontsize=10)
    ax_conv.grid(True, which='both', alpha=0.3)

    # ── (D) Optimal Policy Grid ───────────────────────────
    ax_pol = fig.add_subplot(gs[2, 2:])

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
                            fontsize=14, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax_pol.text(c, r, "G\n(Goal)", ha='center', va='center',
                            fontsize=14, color='white', fontweight='bold', zorder=3)
            else:
                a = policy_opt[r, c]
                # show all Q* values small, highlight best
                offsets = {0: (-0.28, 0.0), 1: (0.0, 0.28),
                           2: (0.28, 0.0),  3: (0.0, -0.28)}
                for ai in range(N_ACTIONS):
                    ox, oy = offsets[ai]
                    is_best = (ai == a)
                    ax_pol.text(c + ox, r + oy,
                                f"{ACTION_ARROWS[ai]}{Q_opt[r,c,ai]:.3f}",
                                ha='center', va='center', fontsize=10,
                                fontweight='bold' if is_best else 'normal',
                                color='#B71C1C' if is_best else '#555',
                                zorder=3)

    ax_pol.set_xlim(-0.5, WIDTH - 0.5)
    ax_pol.set_ylim(HEIGHT - 0.5, -0.5)
    ax_pol.set_xticks(range(WIDTH))
    ax_pol.set_xticklabels([f"col{c}" for c in range(WIDTH)], fontsize=13)
    ax_pol.set_yticks(range(HEIGHT))
    ax_pol.set_yticklabels([f"row{r}" for r in range(HEIGHT)], fontsize=13)
    ax_pol.set_title("(D) Optimal Policy  π*(s) = argmax_a q*(s,a)\n"
                     "(all q* values shown  |  red bold = argmax)",
                     fontsize=13, fontweight='bold')
    ax_pol.set_aspect('equal')

    legend_elems = [
        mpatches.Patch(color='#37474F', label='H: Hole  (terminal, fail)'),
        mpatches.Patch(color='#1565C0', label='G: Goal  (terminal, success)'),
        mpatches.Patch(color='#E8F5E9', label='S/F: Safe  (red = π*)'),
    ]
    ax_pol.legend(handles=legend_elems, loc='lower right',
                  fontsize=8, framealpha=0.9)

    plt.savefig("./d_optimal_state_action_values_img.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: d_optimal_state_action_values_img.png")


def main():
    # ── Run both algorithms ──────────────────────────
    Q_exp, delta_exp = policy_evaluation(env, gamma=GAMMA)   # Expectation
    Q_opt, delta_opt = value_iteration(env, gamma=GAMMA)     # Optimality
    policy_opt = extract_policy(env, Q_opt)

    # ── Console output ───────────────────────────────
    print_results(Q_exp, Q_opt, policy_opt)

    # ── Visualization ────────────────────────────────
    visualize(env, Q_exp, Q_opt, delta_exp, delta_opt, policy_opt)

    env.close()


if __name__ == "__main__":
    main()
