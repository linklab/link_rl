"""
정책 반복 (Policy Iteration) 알고리즘 분석
- E-step (정책 평가): Bellman 기대 방정식으로 V_pi 수렴까지 반복
- I-step (정책 향상): argmax_a q_pi(s,a) 로 탐욕 정책 갱신
- 반복마다 E-step 스윕 수, 정책 변화 수, V-error 추적
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA   = 0.99
THETA   = 1e-6          # E-step 수렴 기준
MAX_OUTER = 50          # 최대 정책 반복 횟수
SHOW_ITERS = 5          # 시각화할 최대 반복 횟수

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH = 4, 4
N_STATES, N_ACTIONS = 16, 4
ACTION_SYMBOLS = ['←', '↓', '→', '↑']

# ── 환경 생성 ──────────────────────────────────────────────────
env = gym.make('FrozenLake-v1',
               desc=MAP_4x4,
               is_slippery=False,
               render_mode=None)

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s 의 타일 종류 반환"""
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]

def policy_evaluation(env, policy, gamma=GAMMA, theta=THETA):
    """
    E-step: 현재 정책 policy 에 대한 가치 함수를 반복적으로 계산.
    반환: V (np.ndarray, shape=N_STATES), sweep 횟수
    """
    V = np.zeros(N_STATES)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(N_STATES):
            if get_tile(s) in ('H', 'G'):
                continue
            a = policy[s]
            v_new = sum(p * (r + gamma * V[s_])
                        for p, s_, r, _ in env.unwrapped.P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        sweeps += 1
        if delta < theta:
            break
    return V, sweeps

def policy_improvement(env, V, gamma=GAMMA):
    """
    I-step: V 에 기반한 탐욕 정책 산출.
    반환: new_policy (np.ndarray, shape=N_STATES), 변화된 상태 수
    """
    new_policy = np.zeros(N_STATES, dtype=int)
    changes = 0
    for s in range(N_STATES):
        if get_tile(s) in ('H', 'G'):
            continue
        q = [sum(p * (r + gamma * V[s_])
                 for p, s_, r, _ in env.unwrapped.P[s][a])
             for a in range(N_ACTIONS)]
        new_policy[s] = int(np.argmax(q))
    return new_policy

def policy_iteration(env, gamma=GAMMA, theta=THETA, max_outer=MAX_OUTER):
    """
    정책 반복 메인 루프.
    반환:
        V_history       : 각 반복 후 V (list of np.ndarray)
        policy_history  : 각 반복 후 policy (list of np.ndarray)
        sweeps_per_iter : E-step 스윕 수 (list of int)
        changes_per_iter: I-step 정책 변화 수 (list of int)
        v_error_trace   : 매 스윕마다 기록한 ||V - V_prev||_inf (list)
        boundary_sweeps : 각 반복이 끝나는 누적 스윕 인덱스 (list of int)
    """
    policy = np.zeros(N_STATES, dtype=int)   # 초기 정책: 모두 행동 0
    V      = np.zeros(N_STATES)

    V_history        = []
    policy_history   = []
    sweeps_per_iter  = []
    changes_per_iter = []
    v_error_trace    = []   # 매 스윕 후 최대 변화량
    boundary_sweeps  = []
    cumulative_sweeps = 0

    for _ in range(max_outer):
        # ── E-step: 정책 평가 (스윕별 delta 기록) ─────────────────
        sweep_count = 0
        while True:
            delta = 0.0
            V_prev = V.copy()
            for s in range(N_STATES):
                if get_tile(s) in ('H', 'G'):
                    continue
                a = policy[s]
                v_new = sum(p * (r + gamma * V[s_])
                            for p, s_, r, _ in env.unwrapped.P[s][a])
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            sweep_count += 1
            cumulative_sweeps += 1
            v_error_trace.append(np.max(np.abs(V - V_prev)))
            if delta < theta:
                break
        boundary_sweeps.append(cumulative_sweeps)

        # ── I-step: 정책 향상 ──────────────────────────────────────
        new_policy = policy_improvement(env, V, gamma)
        changes = int(np.sum(new_policy != policy))

        V_history.append(V.copy())
        policy_history.append(new_policy.copy())
        sweeps_per_iter.append(sweep_count)
        changes_per_iter.append(changes)

        policy = new_policy
        if changes == 0:
            break

    print(f"\n[정책 반복 완료]")
    print(f"  총 외부 반복 수 : {len(sweeps_per_iter)}")
    print(f"  총 E-step 스윕  : {sum(sweeps_per_iter)}")
    print(f"  반복별 E-step 스윕: {sweeps_per_iter}")
    print(f"  반복별 정책 변화: {changes_per_iter}")

    return (V_history, policy_history,
            sweeps_per_iter, changes_per_iter,
            v_error_trace, boundary_sweeps)


# ── 시각화 ────────────────────────────────────────────────────
def draw_heatmap(ax, V, policy, title, cmap):
    """V 히트맵 + 정책 화살표"""
    grid = V.reshape(HEIGHT, WIDTH)
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=V.max() + 1e-8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    for s in range(N_STATES):
        r, c = divmod(s, WIDTH)
        tile  = get_tile(s)
        color = 'white' if V[s] > V.max() * 0.5 else 'black'
        if tile == 'H':
            ax.text(c, r, 'H', ha='center', va='center',
                    fontsize=11, color='gray', fontweight='bold')
        elif tile == 'G':
            ax.text(c, r, 'G', ha='center', va='center',
                    fontsize=11, color='gold', fontweight='bold')
        else:
            ax.text(c, r, f'{V[s]:.3f}', ha='center', va='center',
                    fontsize=7, color=color)
            ax.text(c, r + 0.35, ACTION_SYMBOLS[policy[s]],
                    ha='center', va='center', fontsize=10, color='cyan')
    return im


def visualize(V_history, policy_history,
              sweeps_per_iter, changes_per_iter,
              v_error_trace, boundary_sweeps):

    n_iters  = len(V_history)
    show_n   = min(n_iters, SHOW_ITERS)
    cmap     = plt.colormaps['YlOrRd'].copy()
    cmap.set_bad('lightgray')

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Policy Iteration  —  E-step (Policy Evaluation) + I-step (Policy Improvement)\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  "
        f"θ={THETA}  |  Total outer iters: {n_iters}  |  "
        f"Total sweeps: {sum(sweeps_per_iter)}",
        fontsize=12, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, show_n,
                           top=0.90, bottom=0.07,
                           hspace=0.55, wspace=0.35)

    # ── Row 0: 반복별 V 히트맵 ─────────────────────────────────
    for i in range(show_n):
        ax = fig.add_subplot(gs[0, i])
        sw = sweeps_per_iter[i]
        ch = changes_per_iter[i]
        label = "π*" if changes_per_iter[i] == 0 else f"π{i+1}"
        title = (f"Iter {i+1}  ({label})\n"
                 f"E-sweeps: {sw}   Policy Δ: {ch}")
        im = draw_heatmap(ax, V_history[i], policy_history[i], title, cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 1: E-step 스윕 수 & 정책 변화 수 ──────────────────
    ax_sw = fig.add_subplot(gs[1, :show_n // 2 + show_n % 2])
    ax_ch = fig.add_subplot(gs[1, show_n // 2 + show_n % 2:])

    x = np.arange(1, n_iters + 1)
    bars = ax_sw.bar(x, sweeps_per_iter, color='steelblue', edgecolor='navy', alpha=0.85)
    ax_sw.bar_label(bars, fontsize=8)
    ax_sw.set_xlabel("Policy Iteration #", fontsize=9)
    ax_sw.set_ylabel("E-step Sweeps", fontsize=9)
    ax_sw.set_title("(B) E-step Sweeps per Outer Iteration\n"
                    "(Early iters need many sweeps; later iters converge quickly)",
                    fontsize=9, fontweight='bold')
    ax_sw.set_xticks(x)
    ax_sw.set_ylim(0, max(sweeps_per_iter) * 1.25)
    ax_sw.grid(axis='y', alpha=0.4)

    bars2 = ax_ch.bar(x, changes_per_iter, color='tomato', edgecolor='darkred', alpha=0.85)
    ax_ch.bar_label(bars2, fontsize=8)
    ax_ch.set_xlabel("Policy Iteration #", fontsize=9)
    ax_ch.set_ylabel("Policy Changes (# states)", fontsize=9)
    ax_ch.set_title("(C) Policy Changes per I-step\n"
                    "(0 = no change → optimal policy π* reached)",
                    fontsize=9, fontweight='bold')
    ax_ch.set_xticks(x)
    ax_ch.set_ylim(0, max(changes_per_iter) * 1.3 + 1)
    ax_ch.grid(axis='y', alpha=0.4)

    # ── Row 2: 스윕별 V 변화량 (수렴 곡선) ────────────────────
    ax_conv = fig.add_subplot(gs[2, :])
    sweeps_x = np.arange(1, len(v_error_trace) + 1)
    ax_conv.semilogy(sweeps_x, v_error_trace,
                     color='royalblue', linewidth=1.5, label='||ΔV||∞ per sweep')

    colors_bound = plt.cm.tab10.colors
    for i, b in enumerate(boundary_sweeps):
        col = colors_bound[i % 10]
        ax_conv.axvline(x=b, color=col, linestyle='--', linewidth=1.2, alpha=0.8)
        ax_conv.text(b, ax_conv.get_ylim()[0] * 1.5,
                     f'Iter{i+1}\nend', ha='right', va='bottom',
                     fontsize=7, color=col)

    ax_conv.axhline(y=THETA, color='red', linestyle=':', linewidth=1,
                    label=f'θ = {THETA}  (convergence threshold)')
    ax_conv.set_xlabel("Cumulative E-step Sweeps", fontsize=9)
    ax_conv.set_ylabel("||ΔV||∞  (log scale)", fontsize=9)
    ax_conv.set_title(
        "(D) Convergence of V over Every E-step Sweep\n"
        "(Dashed lines = policy iteration boundary; "
        "early iters require many sweeps to converge policy eval)",
        fontsize=9, fontweight='bold')
    ax_conv.legend(fontsize=8, loc='upper right')
    ax_conv.grid(True, alpha=0.3)

    plt.savefig('./a_policy_iteration_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] a_policy_iteration.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  정책 반복 (Policy Iteration) 알고리즘 분석")
    print("  E-step: 정책 평가  |  I-step: 정책 향상")
    print("=" * 60)

    (V_history, policy_history,
     sweeps_per_iter, changes_per_iter,
     v_error_trace, boundary_sweeps) = policy_iteration(env)

    visualize(V_history, policy_history,
              sweeps_per_iter, changes_per_iter,
              v_error_trace, boundary_sweeps)

    env.close()
