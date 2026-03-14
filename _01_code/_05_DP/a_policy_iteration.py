"""
정책 반복 (Policy Iteration) 알고리즘 분석
- E-step (정책 평가): Bellman 기대 방정식으로 V_pi 수렴까지 반복
- I-step (정책 향상): argmax_a q_pi(s,a) 로 탐욕 정책 갱신
- 반복마다 E-step 스윕 수, 정책 변화 수, V-error 추적
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA   = 0.99
THETA   = 1e-9          # E-step 수렴 기준
MAX_OUTER = 50          # 최대 정책 반복 횟수
SHOW_ITERS = 5          # 시각화할 최대 반복 횟수

MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
HEIGHT, WIDTH = 4, 4
N_STATES, N_ACTIONS = 16, 4
ACTION_SYMBOLS = ['←', '↓', '→', '↑']

# ── 환경 생성 ──────────────────────────────────────────────────
env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=False, render_mode=None)

# ── 유틸리티 ──────────────────────────────────────────────────
def get_tile(s):
    """상태 s 의 타일 종류 반환"""
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]

def policy_evaluation(env, policy, V=None, gamma=GAMMA, theta=THETA):
    """
    E-step: 현재 정책 policy 에 대한 가치 함수를 반복적으로 계산.
    반환: V (np.ndarray, shape=N_STATES), sweep 횟수, 스윕별 ||ΔV||∞ 목록
    """
    V = np.zeros(N_STATES) if V is None else V.copy()
    sweeps = 0
    sweep_errors = []
    while True:
        delta = 0.0
        V_prev = V.copy()
        for s in range(N_STATES):
            if get_tile(s) in ('H', 'G'):
                continue
            a = policy[s]
            v_new = sum(
                p * (r + gamma * V[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]
            )
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        sweeps += 1
        sweep_errors.append(np.max(np.abs(V - V_prev)))
        if delta < theta:
            break
    return V, sweeps, sweep_errors

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
        q = [
            sum(
                p * (r + gamma * V[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]
            ) for a in range(N_ACTIONS)]
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
        delta_trace     : 매 스윕마다 기록한 ||V - V_prev||_inf (list)
        cumulative_sweeps_list : 각 반복이 끝나는 누적 스윕 인덱스 (list of int)
    """
    policy = np.zeros(N_STATES, dtype=int)   # 초기 정책: 모두 행동 0
    V      = np.zeros(N_STATES)

    V_history        = []
    policy_history   = []
    sweeps_per_iter  = []
    changes_per_iter = []
    delta_trace      = []   # 매 스윕 후 최대 변화량
    cumulative_sweeps_list  = []
    cumulative_sweeps = 0

    for _ in range(max_outer):
        # ── E-step: 정책 평가 ──────────────────────────────────────
        V, sweeps, sweep_errors = policy_evaluation(env, policy, V, gamma, theta)
        delta_trace.extend(sweep_errors)
        cumulative_sweeps += sweeps
        cumulative_sweeps_list.append(cumulative_sweeps)

        # ── I-step: 정책 향상 ──────────────────────────────────────
        new_policy = policy_improvement(env, V, gamma)
        changes = int(np.sum(new_policy != policy))

        V_history.append(V.copy())
        policy_history.append(new_policy.copy())
        sweeps_per_iter.append(sweeps)
        changes_per_iter.append(changes)

        policy = new_policy
        if changes == 0:
            break

    print(f"\n[정책 반복 완료]")
    print(f"  외부 반복 수 : {len(sweeps_per_iter)}")
    print(f"  외부 반복별 E-step 스윕: {sweeps_per_iter} = {sum(sweeps_per_iter)}")
    print(f"  외부 반복별 정책 변화 상태 수: {changes_per_iter}")

    return (V_history, policy_history,
            sweeps_per_iter, changes_per_iter,
            delta_trace, cumulative_sweeps_list)


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
              sweeps_per_iter, changes_per_iter):

    n_iters  = len(V_history)
    show_n   = n_iters
    cmap     = plt.colormaps['YlOrRd'].copy()
    cmap.set_bad('lightgray')

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"Policy Iteration  —  E-step (Policy Evaluation) + I-step (Policy Improvement)\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  "
        f"θ={THETA}  |  Total outer iters: {n_iters}  |  "
        f"Total sweeps: {sum(sweeps_per_iter)}",
        fontsize=12, fontweight='bold', y=1.02
    )

    gs = gridspec.GridSpec(1, show_n,
                           top=0.88, bottom=0.07,
                           hspace=0.55, wspace=0.35)

    # ── 반복별 V 히트맵 ────────────────────────────────────────
    for i in range(show_n):
        ax = fig.add_subplot(gs[0, i])
        sw = sweeps_per_iter[i]
        ch = changes_per_iter[i]
        label = "π*" if changes_per_iter[i] == 0 else f"π{i+1}"
        title = (f"Iter {i+1}  ({label})\n"
                 f"E-sweeps: {sw}   Policy Δ: {ch}")
        im = draw_heatmap(ax, V_history[i], policy_history[i], title, cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig('./a_policy_iteration_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] a_policy_iteration_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  정책 반복 (Policy Iteration) 알고리즘 분석")
    print("  E-step: 정책 평가  |  I-step: 정책 향상")
    print("=" * 60)

    (V_history, policy_history,
     sweeps_per_iter, changes_per_iter,
     delta_trace, cumulative_sweeps_list) = policy_iteration(env)

    visualize(V_history, policy_history,
              sweeps_per_iter, changes_per_iter)

    env.close()
