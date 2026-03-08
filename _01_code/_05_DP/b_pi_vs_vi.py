"""
정책 반복 (PI) vs 가치 반복 (VI) 비교 분석
- 수렴 속도 (스윕당 V-error 감소)
- 계산 비용 (총 스윕 수)
- 최종 V* 품질 비교
- 할인율 γ 에 따른 민감도
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

# ── 하이퍼파라미터 ─────────────────────────────────────────────
GAMMA       = 0.99
THETA_EVAL  = 1e-6    # PI E-step 수렴 기준
THETA_VI    = 1e-6    # VI 수렴 기준
THETA_REF   = 1e-12   # 기준 V* 계산용 (고정밀도)
GAMMA_LIST  = [0.70, 0.80, 0.90, 0.95, 0.99]   # γ 민감도 분석

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
    r, c = divmod(s, WIDTH)
    return MAP_4x4[r][c]

def compute_v_star(env, gamma, theta=THETA_REF):
    """고정밀 V* 계산 (VI로 기준값 생성)"""
    V = np.zeros(N_STATES)
    while True:
        delta = 0.0
        for s in range(N_STATES):
            if get_tile(s) in ('H', 'G'):
                continue
            q = [sum(p * (r + gamma * V[s_])
                     for p, s_, r, _ in env.unwrapped.P[s][a])
                 for a in range(N_ACTIONS)]
            v_new = max(q)
            delta = max(delta, abs(v_new - V[s]))
            V[s]  = v_new
        if delta < theta:
            break
    return V

def greedy_policy(env, V, gamma):
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if get_tile(s) in ('H', 'G'):
            continue
        q = [sum(p * (r + gamma * V[s_])
                 for p, s_, r, _ in env.unwrapped.P[s][a])
             for a in range(N_ACTIONS)]
        policy[s] = int(np.argmax(q))
    return policy

# ── PI ─────────────────────────────────────────────────────────
def run_policy_iteration(env, V_star, gamma=GAMMA,
                          theta_eval=THETA_EVAL):
    """
    정책 반복 실행.
    반환:
        V_final         : 최적 가치 함수
        policy_final    : 최적 정책
        v_error_trace   : 매 스윕마다 ||V - V*||_inf
        sweeps_per_iter : 외부 반복별 E-step 스윕 수
        boundary_sweeps : 외부 반복 경계 (누적 스윕 인덱스)
    """
    policy = np.zeros(N_STATES, dtype=int)
    V      = np.zeros(N_STATES)
    v_error_trace   = []
    sweeps_per_iter = []
    boundary_sweeps = []
    cumulative      = 0

    while True:
        # E-step
        sweep_count = 0
        while True:
            delta = 0.0
            for s in range(N_STATES):
                if get_tile(s) in ('H', 'G'):
                    continue
                a    = policy[s]
                v_new = sum(p * (r + gamma * V[s_])
                            for p, s_, r, _ in env.unwrapped.P[s][a])
                delta = max(delta, abs(v_new - V[s]))
                V[s]  = v_new
            sweep_count += 1
            cumulative  += 1
            v_error_trace.append(float(np.max(np.abs(V - V_star))))
            if delta < theta_eval:
                break
        sweeps_per_iter.append(sweep_count)
        boundary_sweeps.append(cumulative)

        # I-step
        new_policy = greedy_policy(env, V, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    print(f"\n[PI 완료]  외부 반복: {len(sweeps_per_iter)}  "
          f"총 스윕: {sum(sweeps_per_iter)}  "
          f"최종 V-error: {v_error_trace[-1]:.2e}")
    return V, policy, v_error_trace, sweeps_per_iter, boundary_sweeps

# ── VI ─────────────────────────────────────────────────────────
def run_value_iteration(env, V_star, gamma=GAMMA,
                         theta=THETA_VI):
    """
    가치 반복 실행.
    반환:
        V_final       : 최적 가치 함수
        policy_final  : 탐욕 정책
        v_error_trace : 매 스윕마다 ||V - V*||_inf
        delta_trace   : 매 스윕마다 max Bellman residual
    """
    V = np.zeros(N_STATES)
    v_error_trace = []
    delta_trace   = []

    while True:
        delta = 0.0
        for s in range(N_STATES):
            if get_tile(s) in ('H', 'G'):
                continue
            q     = [sum(p * (r + gamma * V[s_])
                         for p, s_, r, _ in env.unwrapped.P[s][a])
                     for a in range(N_ACTIONS)]
            v_new = max(q)
            delta = max(delta, abs(v_new - V[s]))
            V[s]  = v_new
        v_error_trace.append(float(np.max(np.abs(V - V_star))))
        delta_trace.append(delta)
        if delta < theta:
            break

    policy = greedy_policy(env, V, gamma)
    print(f"[VI 완료]  총 스윕: {len(v_error_trace)}  "
          f"최종 V-error: {v_error_trace[-1]:.2e}")
    return V, policy, v_error_trace, delta_trace

# ── γ 민감도 ────────────────────────────────────────────────────
def gamma_sensitivity(env, gamma_list=GAMMA_LIST):
    """각 γ 에 대해 PI, VI 의 총 스윕 수 비교"""
    results = {}
    for g in gamma_list:
        V_star = compute_v_star(env, g)
        _, _, _, sw_pi, _ = run_policy_iteration(env, V_star, gamma=g)
        _, _, err_vi, _   = run_value_iteration(env, V_star, gamma=g)
        results[g] = dict(pi_sweeps=sum(sw_pi),
                          vi_sweeps=len(err_vi))
    return results

# ── 시각화 ────────────────────────────────────────────────────
def draw_heatmap(ax, V, policy, title, cmap):
    grid = V.reshape(HEIGHT, WIDTH)
    im   = ax.imshow(grid, cmap=cmap, vmin=0, vmax=V.max() + 1e-8)
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


def visualize(V_pi, policy_pi, v_error_pi, sweeps_pi, boundary_pi,
              V_vi, policy_vi, v_error_vi, gamma_results):

    cmap = plt.colormaps['YlOrRd'].copy()
    n_pi_sweeps = sum(sweeps_pi)
    n_vi_sweeps = len(v_error_vi)

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Policy Iteration (PI)  vs.  Value Iteration (VI)\n"
        f"FrozenLake 4×4  |  is_slippery=False  |  γ={GAMMA}",
        fontsize=13, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(3, 6,
                           top=0.93, bottom=0.07,
                           hspace=0.55, wspace=0.45)

    # ── (A) 수렴 곡선 비교 ──────────────────────────────────────
    ax_conv = fig.add_subplot(gs[0, :4])

    # PI 곡선
    x_pi = np.arange(1, n_pi_sweeps + 1)
    ax_conv.semilogy(x_pi, v_error_pi,
                     color='steelblue', linewidth=2,
                     label=f'PI  (total {n_pi_sweeps} sweeps)')

    # PI 외부 반복 경계 표시
    for i, b in enumerate(boundary_pi):
        ax_conv.axvline(x=b, color='steelblue', linestyle='--',
                        linewidth=0.9, alpha=0.6)
        ax_conv.text(b + 0.3, ax_conv.get_ylim()[0],
                     f'i{i+1}', fontsize=6, color='steelblue', va='bottom')

    # VI 곡선 (x축을 PI 스윕 수 기준으로 매핑)
    x_vi = np.linspace(1, n_pi_sweeps, n_vi_sweeps)
    ax_conv.semilogy(x_vi, v_error_vi,
                     color='tomato', linewidth=2,
                     label=f'VI  (total {n_vi_sweeps} sweeps)')

    ax_conv.axhline(y=THETA_EVAL, color='gray', linestyle=':',
                    linewidth=1, label=f'θ = {THETA_EVAL}')
    ax_conv.set_xlabel("Sweep #  (PI: actual sweeps / VI: rescaled to same axis)",
                       fontsize=9)
    ax_conv.set_ylabel("||V - V*||∞  (log scale)", fontsize=9)
    ax_conv.set_title(
        "(A) Convergence Comparison: PI vs. VI\n"
        "PI dashed lines = outer iteration boundary (E-step end + I-step)",
        fontsize=9, fontweight='bold')
    ax_conv.legend(fontsize=8)
    ax_conv.grid(True, alpha=0.3)

    # ── (B) 총 스윕 수 막대 ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 4:])
    bars = ax_bar.bar(['PI', 'VI'],
                      [n_pi_sweeps, n_vi_sweeps],
                      color=['steelblue', 'tomato'],
                      edgecolor='black', alpha=0.85, width=0.5)
    ax_bar.bar_label(bars, fontsize=12, fontweight='bold', padding=3)
    ax_bar.set_ylabel("Total Sweeps", fontsize=9)
    ax_bar.set_title("(B) Total Sweep Count\n(Lower = less computation)",
                     fontsize=9, fontweight='bold')
    ax_bar.set_ylim(0, max(n_pi_sweeps, n_vi_sweeps) * 1.3)
    ax_bar.grid(axis='y', alpha=0.4)

    # 주석: 어느 쪽이 빠른지
    winner = 'VI' if n_vi_sweeps < n_pi_sweeps else 'PI'
    ratio  = max(n_pi_sweeps, n_vi_sweeps) / max(min(n_pi_sweeps, n_vi_sweeps), 1)
    ax_bar.text(0.5, 0.88,
                f'{winner} is faster\n(×{ratio:.1f})',
                ha='center', va='top', transform=ax_bar.transAxes,
                fontsize=9, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                          edgecolor='darkgreen', alpha=0.8))

    # ── (C)(D) 최종 V* 히트맵 비교 ──────────────────────────────
    ax_pi = fig.add_subplot(gs[1, :3])
    ax_vi = fig.add_subplot(gs[1, 3:])

    im_pi = draw_heatmap(ax_pi, V_pi, policy_pi,
                         f"(C) PI  —  Final V*  (after {n_pi_sweeps} sweeps)", cmap)
    im_vi = draw_heatmap(ax_vi, V_vi, policy_vi,
                         f"(D) VI  —  Final V*  (after {n_vi_sweeps} sweeps)", cmap)
    plt.colorbar(im_pi, ax=ax_pi, fraction=0.046, pad=0.04)
    plt.colorbar(im_vi, ax=ax_vi, fraction=0.046, pad=0.04)

    # ── (E) PI E-step 스윕 수 막대 ────────────────────────────────
    ax_sw = fig.add_subplot(gs[2, :3])
    x_iters = np.arange(1, len(sweeps_pi) + 1)
    b2 = ax_sw.bar(x_iters, sweeps_pi,
                   color='steelblue', edgecolor='navy', alpha=0.85)
    ax_sw.bar_label(b2, fontsize=8)
    ax_sw.set_xlabel("PI Outer Iteration #", fontsize=9)
    ax_sw.set_ylabel("E-step Sweeps", fontsize=9)
    ax_sw.set_title(
        "(E) PI: E-step Sweeps per Outer Iteration\n"
        "PI drawback: early iters require many eval sweeps",
        fontsize=9, fontweight='bold')
    ax_sw.set_xticks(x_iters)
    ax_sw.set_ylim(0, max(sweeps_pi) * 1.3)
    ax_sw.grid(axis='y', alpha=0.4)

    # ── (F) γ 민감도 ─────────────────────────────────────────────
    ax_gm = fig.add_subplot(gs[2, 3:])
    gammas     = list(gamma_results.keys())
    pi_totals  = [gamma_results[g]['pi_sweeps'] for g in gammas]
    vi_totals  = [gamma_results[g]['vi_sweeps'] for g in gammas]

    x_g = np.arange(len(gammas))
    w   = 0.35
    b_pi = ax_gm.bar(x_g - w/2, pi_totals, width=w,
                     color='steelblue', edgecolor='navy',
                     alpha=0.85, label='PI')
    b_vi = ax_gm.bar(x_g + w/2, vi_totals, width=w,
                     color='tomato', edgecolor='darkred',
                     alpha=0.85, label='VI')
    ax_gm.bar_label(b_pi, fontsize=7, rotation=45, padding=2)
    ax_gm.bar_label(b_vi, fontsize=7, rotation=45, padding=2)
    ax_gm.set_xticks(x_g)
    ax_gm.set_xticklabels([f'γ={g}' for g in gammas], fontsize=8)
    ax_gm.set_ylabel("Total Sweeps", fontsize=9)
    ax_gm.set_title(
        "(F) Discount Factor γ Sensitivity\n"
        "Higher γ → longer horizon → more sweeps needed",
        fontsize=9, fontweight='bold')
    ax_gm.legend(fontsize=8)
    ax_gm.grid(axis='y', alpha=0.4)

    plt.savefig('./b_pi_vs_vi_img.png',
                dpi=130, bbox_inches='tight')
    print("\n[시각화 저장] b_pi_vs_vi.png")
    plt.show()


# ── 결과 요약 출력 ─────────────────────────────────────────────
def print_summary(V_pi, V_vi, v_error_pi, v_error_vi,
                  sweeps_pi, gamma_results):
    print("\n" + "=" * 55)
    print("  PI vs VI 비교 요약 (γ = {:.2f})".format(GAMMA))
    print("=" * 55)
    print(f"  {'항목':<30} {'PI':>10} {'VI':>10}")
    print(f"  {'-'*50}")
    print(f"  {'총 스윕 수':<30} {sum(sweeps_pi):>10} {len(v_error_vi):>10}")
    print(f"  {'외부 반복 수':<30} {len(sweeps_pi):>10} {'N/A':>10}")
    print(f"  {'최종 V-error (||V-V*||∞)':<30} {v_error_pi[-1]:>10.2e} {v_error_vi[-1]:>10.2e}")
    print(f"  {'V* 최대 차이 |V_PI - V_VI|':<30} {np.max(np.abs(V_pi - V_vi)):>10.2e}")
    print()
    print("  γ 민감도 (총 스윕 수):")
    print(f"  {'γ':<10} {'PI 스윕':>10} {'VI 스윕':>10}")
    for g, r in gamma_results.items():
        print(f"  {g:<10.2f} {r['pi_sweeps']:>10} {r['vi_sweeps']:>10}")
    print()
    print("  [특징 비교]")
    print("  PI : 정책 고정 후 V 완전 수렴 → 정책 갱신 (E-step 비용 큼)")
    print("  VI : 매 스윕 Bellman 최적 방정식 적용 → 암묵적 정책 향상 (스윕 단순)")


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  정책 반복 (PI) vs 가치 반복 (VI) 비교 분석")
    print("=" * 60)

    # 기준 V* 계산
    print("\n기준 V* 계산 중 (고정밀도)...")
    V_star = compute_v_star(env, GAMMA)

    # PI 실행
    print("\n--- Policy Iteration ---")
    V_pi, policy_pi, v_error_pi, sweeps_pi, boundary_pi = \
        run_policy_iteration(env, V_star)

    # VI 실행
    print("\n--- Value Iteration ---")
    V_vi, policy_vi, v_error_vi, _ = \
        run_value_iteration(env, V_star)

    # γ 민감도
    print("\n--- γ 민감도 분석 ---")
    gamma_results = gamma_sensitivity(env)

    # 요약 출력
    print_summary(V_pi, V_vi, v_error_pi, v_error_vi,
                  sweeps_pi, gamma_results)

    # 시각화
    visualize(V_pi, policy_pi, v_error_pi, sweeps_pi, boundary_pi,
              V_vi, policy_vi, v_error_vi, gamma_results)

    env.close()
