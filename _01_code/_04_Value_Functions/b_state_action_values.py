"""
Frozen Lake (is_slippery=False) — State-Action Value Function (Q-function)
Iterative Policy Evaluation using Bellman Expectation Equation for Q

Environment:
  - 4x4 Frozen Lake,  is_slippery=False  (deterministic)
  - Uniform random policy: pi(a|s) = 0.25 for all actions
  - Discount factor gamma = 0.99

Bellman Expectation Equation for Q (deterministic, p(s'|s,a)=1):
  q_pi(s,a) = r + gamma * sum_{a'} pi(a'|s') * q_pi(s', a')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ══════════════════════════════════════════════════
# 1. 환경 설정
# ══════════════════════════════════════════════════
MAP_4x4 = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]

GAMMA   = 0.99   # 감가율
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
# 2. 반복 정책 평가 (Iterative Policy Evaluation)
#
#  v(s) ← Σ_a π(a|s) * Σ_{s'} p(s'|s,a) * [r + γ·v(s')]
#  결정적 환경이므로:
#  v(s) ← Σ_a π(a|s) * [r(s,a) + γ·v(s')]
# ══════════════════════════════════════════════════
def iterative_policy_evaluation(env, gamma=GAMMA, theta=1e-9, max_iter=100000):
    """
    Bellman Expectation Equation for Q-function (state-action value)

    q_pi(s,a) = sum_{s'} p(s'|s,a) * [r + gamma * sum_{a'} pi(a'|s') * q_pi(s',a')]

    For deterministic env (p=1.0):
      q_pi(s,a) = r + gamma * sum_{a'} pi(a'|s') * q_pi(s', a')

    P[s][a] = [(prob, next_state, reward, terminated), ...]
    Returns Q : shape (N_STATES, N_ACTIONS)
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    delta_history = []

    for iteration in range(max_iter):
        delta = 0.0
        Q_new = np.copy(Q)

        for s in range(N_STATES):
            row, col = divmod(s, WIDTH)

            # Terminal states (H, G): Q = 0 fixed
            if CELL[(row, col)] in ('H', 'G'):
                Q_new[s, :] = 0.0
                continue

            for a in range(N_ACTIONS):
                q_new = 0.0
                for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                    # V_pi(s') = sum_{a'} pi(a'|s') * Q(s',a')  (policy evaluation)
                    v_next = PI * np.sum(Q[next_s, :])
                    q_new += prob * (reward + gamma * v_next)

                delta         = max(delta, abs(Q[s, a] - q_new))
                Q_new[s, a]   = q_new

        Q = Q_new
        delta_history.append(delta)

        if delta < theta:
            print(f"\nConverged at iteration {iteration+1}  "
                  f"(Delta={delta:.2e} < theta={theta:.0e})")
            break
    else:
        print(f"Reached max iterations {max_iter}  (final Delta={delta:.2e})")

    return Q.reshape(HEIGHT, WIDTH, N_ACTIONS), delta_history


def print_value_function(Q_grid):
    """Q_grid shape: (HEIGHT, WIDTH, N_ACTIONS)"""
    print(f"\n  Converged Q_pi(s,a)  (gamma={GAMMA}, uniform random policy pi=0.25)")
    print(f"  Action: 0=LEFT  1=DOWN  2=RIGHT  3=UP")
    print(f"  {'State':<12}  {'LEFT':>8}  {'DOWN':>8}  {'RIGHT':>8}  {'UP':>8}  {'V_pi=mean':>10}")
    print("  " + "-" * 62)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            s_label = f"({r},{c})[{cell}]"
            if cell in ('H', 'G'):
                print(f"  {s_label:<12}  {'terminal':>8}  {'':>8}  {'':>8}  {'':>8}")
            else:
                q = Q_grid[r, c, :]          # shape (4,)
                v = PI * np.sum(q)           # V_pi = sum_a pi(a|s)*Q(s,a)
                print(f"  {s_label:<12}  "
                      f"{q[0]:>8.4f}  {q[1]:>8.4f}  {q[2]:>8.4f}  {q[3]:>8.4f}  "
                      f"{v:>10.4f}")


def visualize(env, Q_grid, delta_hist):
    # ══════════════════════════════════════════════════
    # 3. Visualization
    # ══════════════════════════════════════════════════
    ACTION_LABELS = ["L", "D", "R", "U"]
    ACTION_ARROWS = ["←", "↓", "→", "↑"]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"Frozen Lake 4×4  (is_slippery=False)\n"
        f"Uniform Random Policy  pi=0.25 | Discount Factor gamma={GAMMA} | "
        f"Bellman Expectation Equation for Q(s,a)",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.40, wspace=0.35,
                           width_ratios=[1.6, 1])

    # ────────────────────────────────────────────────
    # (C) Cell-level Q-value table  (all 4 actions in one grid)
    # ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 0])
    ax3.set_xlim(-0.5, WIDTH - 0.5)
    ax3.set_ylim(HEIGHT - 0.5, -0.5)
    ax3.set_xticks(range(WIDTH))
    ax3.set_xticklabels([f"col{c}" for c in range(WIDTH)], fontsize=13)
    ax3.set_yticks(range(HEIGHT))
    ax3.set_yticklabels([f"row{r}" for r in range(HEIGHT)], fontsize=13)
    ax3.set_title("(C) Q_pi(s,a) — All Actions per Cell\n"
                  "(top=U, bottom=D, left=L, right=R  |  bold red=argmax)",
                  fontsize=13, fontweight='bold')

    # cell background
    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            bg = '#37474F' if cell == 'H' else ('#1565C0' if cell == 'G' else '#F9FBE7')
            ax3.add_patch(plt.Rectangle(
                (c-.5, r-.5), 1, 1, facecolor=bg, zorder=1,
                edgecolor='#888', linewidth=1.8))

    # Q values inside each cell
    offsets = {0: (-0.32, 0.0),   # LEFT
               1: (0.0,   0.30),  # DOWN
               2: (0.32,  0.0),   # RIGHT
               3: (0.0,  -0.30)}  # UP

    for r in range(HEIGHT):
        for c in range(WIDTH):
            cell = CELL[(r, c)]
            if cell == 'H':
                ax3.text(c, r, "H\n(Hole)", ha='center', va='center',
                         fontsize=14, color='white', fontweight='bold', zorder=3)
            elif cell == 'G':
                ax3.text(c, r, "G\n(Goal)", ha='center', va='center',
                         fontsize=14, color='white', fontweight='bold', zorder=3)
            else:
                q = Q_grid[r, c, :]
                best_a = int(np.argmax(q))
                for a_idx in range(N_ACTIONS):
                    ox, oy = offsets[a_idx]
                    is_best = (a_idx == best_a)
                    ax3.text(c + ox, r + oy,
                             f"{ACTION_ARROWS[a_idx]}{q[a_idx]:.3f}",
                             ha='center', va='center',
                             fontsize=11,
                             fontweight='bold' if is_best else 'normal',
                             color='#B71C1C' if is_best else '#333',
                             zorder=3)

    ax3.set_aspect('equal')

    # ────────────────────────────────────────────────
    # (B) Convergence curve
    # ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(delta_hist, color='#1976D2', linewidth=2)
    ax2.axhline(1e-9, color='red', linestyle='--', linewidth=1.2,
                label='theta = 1e-9 (convergence threshold)')
    ax2.axvline(len(delta_hist)-1, color='green', linestyle=':',
                linewidth=1.5, label=f'Converged ({len(delta_hist)} iters)')
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Max Delta  (log scale)", fontsize=11)
    ax2.set_title(f"(B) Convergence Curve\n(Total {len(delta_hist)} iterations)",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    plt.savefig("./b_state_action_values_img.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: b_state_action_values_img.png")


def main():
    # ── Run ──────────────────────────────────────────
    Q_grid, delta_hist = iterative_policy_evaluation(env, gamma=GAMMA)

    # ── Console output ───────────────────────────────
    print_value_function(Q_grid)

    # ── Visualization ────────────────────────────────
    visualize(env, Q_grid, delta_hist)

    env.close()


if __name__ == "__main__":
    main()