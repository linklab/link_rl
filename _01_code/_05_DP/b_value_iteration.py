"""
가치 반복 (Value Iteration) 알고리즘 분석

[핵심 원리]
  1. while 루프 안: Bellman 최적 방정식으로 V만 반복 갱신 → V* 수렴
       V(s) ← max_a Σ P(s'|s,a)[r + γ V(s')]
  2. 루프 종료 후: 수렴된 V* 로 탐욕 정책 단 1회 산출 → π*
       π*(s) = argmax_a q(s,a)

  ※ Policy Iteration과의 차이:
     PI  : E-step(policy_evaluation) ↔ I-step(policy_improvement) 교대 반복
     VI  : V 갱신만 반복 → 수렴 확인 → π* 1회 추출
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA      = 0.99
THETA      = 1e-9       # VI 수렴 기준 (max Bellman residual < θ)
SHOW_ITERS = 5          # 시각화할 히스토리 스냅샷 수

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

def greedy_policy(env, V, gamma=GAMMA):
    """
    I-step: 수렴된 V* 로 최적 탐욕 정책 1회 산출.
    (Value Iteration 에서는 루프 종료 후 단 한 번만 호출)

    반환: policy (np.ndarray, shape=N_STATES)
    """
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if get_tile(s) in ('H', 'G'):
            continue
        q = [
            sum(
                p * (r + gamma * V[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]
            ) for a in range(N_ACTIONS)
        ]
        policy[s] = int(np.argmax(q))
    return policy


def value_iteration(env, gamma=GAMMA, theta=THETA):
    """
    가치 반복 메인 루프.

    [알고리즘]
      while 루프: V 갱신만 반복 (policy 추출 없음)
        ├─ V_prev = V.copy()
        ├─ for s: V(s) ← max_a q(s,a)   ← Bellman 최적 방정식
        ├─ delta = ||V - V_prev||∞
        └─ delta < θ 이면 break (수렴)
      루프 종료: policy_improvement(V*) 1회 → π*

    반환:
        V_history            : 각 스윕 후 V 스냅샷 (list of np.ndarray)
        policy_final         : 수렴된 V* 로 산출한 최적 정책 π* (np.ndarray)
        delta_trace          : 매 스윕 ||V - V_prev||∞ (list of float)
        cumulative_sweeps_list : 누적 스윕 인덱스 (list of int)
    """
    V = np.zeros(N_STATES)

    V_history              = [V.copy()]   # sweep 0: 초기 V
    delta_trace            = []
    cumulative_sweeps_list = []
    cumulative_sweeps      = 0

    # ── V 갱신 루프 (policy 추출 없음) ─────────────────────────
    while True:
        V_prev = V.copy()
        delta  = 0.0

        for s in range(N_STATES):
            if get_tile(s) in ('H', 'G'):
                continue
            q = [
                sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]
                ) for a in range(N_ACTIONS)
            ]
            v_new  = max(q)                      # Bellman 최적 방정식 (VI 핵심)
            delta  = max(delta, abs(v_new - V[s]))
            V[s]   = v_new

        cumulative_sweeps += 1
        cumulative_sweeps_list.append(cumulative_sweeps)
        delta_trace.append(np.max(np.abs(V - V_prev)))
        V_history.append(V.copy())

        if delta < theta:
            break   # V* 수렴

    # ── 수렴 후 최적 정책 단 1회 산출 ──────────────────────────
    policy_final = greedy_policy(env, V, gamma)   # π* = argmax_a q(s,a)

    print(f"\n[가치 반복 완료]")
    print(f"  총 스윕 수       : {len(delta_trace)}")
    print(f"  최종 Bellman δ   : {delta_trace[-1]:.2e}")
    print(f"  policy_improvement 호출 횟수: 1  (수렴 후 단 1회)")

    return (V_history, policy_final,
            delta_trace, cumulative_sweeps_list)


# ── 시각화 ────────────────────────────────────────────────────
def draw_heatmap(ax, V, policy, title, cmap):
    """V 히트맵 + 최적 정책 화살표"""
    grid = V.reshape(HEIGHT, WIDTH)
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=V.max() + 1e-8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    for s in range(N_STATES):
        r, c  = divmod(s, WIDTH)
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


def visualize(V_history, policy_final, delta_trace, cumulative_sweeps_list):
    """
    visualize:
      - V_history 스냅샷: 중간 V 변화 추적 (수렴 과정 확인용)
      - policy_final    : 수렴 후 단 1회 산출된 π* 를 모든 히트맵에 표시
    """
    total_sweeps = len(delta_trace)

    show_V  = V_history[1:]        # Init(sweep 0) 제외, sweep 1~ 전체
    show_sw = cumulative_sweeps_list
    n_show  = len(show_V)

    cmap = plt.colormaps['YlOrRd'].copy()
    cmap.set_bad('lightgray')

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"Value Iteration  —  Converge V* first, then extract π* once\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}  |  "
        f"θ={THETA}  |  Total sweeps: {total_sweeps}",
        fontsize=12, fontweight='bold', y=1.02
    )

    gs = gridspec.GridSpec(1, n_show,
                           top=0.88, bottom=0.07,
                           hspace=0.55, wspace=0.35)

    # ── V 수렴 히스토리 히트맵 (화살표는 최종 π*) ────────────────
    for i, (sw, Vi) in enumerate(zip(show_sw, show_V)):
        ax = fig.add_subplot(gs[0, i])
        is_last = (i == n_show - 1)
        label   = "V*  →  π*" if is_last else f"V_{sw}"
        d_str   = f"δ={delta_trace[sw-1]:.2e}" if sw > 0 else "Init"
        title   = f"Sweep {sw}  ({label})\n{d_str}"
        if is_last:
            title += "  ✓ Converged"
        im = draw_heatmap(ax, Vi, policy_final, title, cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig('./b_value_iteration_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] b_value_iteration_img.png")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  가치 반복 (Value Iteration)")
    print("  [1단계] Bellman 최적 방정식 반복 → V* 수렴")
    print("  [2단계] 수렴 후 policy_improvement 1회 → π*")
    print("=" * 60)

    (V_history, policy_final,
     delta_trace, cumulative_sweeps_list) = value_iteration(env)

    visualize(V_history, policy_final,
              delta_trace, cumulative_sweeps_list)

    env.close()