"""
b_frozen_lake_sarsa_comparison_fixed.py
────────────────────────────────────────────────────────────────────────────────
4가지 TD 제어 알고리즘 비교 (FrozenLake-v1, 4x4, non-slippery)
  1) Q-Learning        (Off-Policy)
  2) SARSA             (On-Policy)
  3) Expected SARSA    (On-Policy)
  4) Off-Policy SARSA  (IS 가중치 적용)

【수정 사항 요약】
  버그 1 - train() 조기 종료로 인한 x축 범위 불일치
    원인: avg_reward == 1.0 달성 시 즉시 is_train_success=True → break
          Q-Learning/SARSA처럼 빠르게 수렴하는 알고리즘은
          validation_curve가 짧은 step 범위에서만 생성됨.
    수정: is_train_success 판단 후에도 학습을 멈추지 않고 num_episodes까지 계속 진행.
          대신 train() 성공 여부 반환은 유지.

  버그 2 - interpolate_curves()의 알고리즘별 x축 범위 불일치
    원인: interpolate_curves(curves)를 알고리즘마다 독립적으로 호출하므로
          max_time_step이 알고리즘마다 달라짐.
          → 같은 plot에 그릴 때 각 알고리즘이 서로 다른 x 구간에 그려짐.
    수정: plot_comparison()에서 모든 알고리즘의 curves를 모아
          global_max_time_step을 먼저 계산하고,
          동일한 x_values를 interpolate_curves()에 전달.
────────────────────────────────────────────────────────────────────────────────
"""

import random
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from a_frozen_lake_q_learning import QTableAgent, MAP_NAME, DESC

IS_SLIPPERY = False

# ════════════════════════════════════════════════════════════════════════════════
# Base Agent (QTableAgent 재사용)
# ════════════════════════════════════════════════════════════════════════════════
class BaseComparisonAgent(QTableAgent):
    def __init__(self, config):
        env = gym.make("FrozenLake-v1", desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
        super().__init__(env=env, config=config, use_wandb=False)

    def train(self):
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════════
# 1) Q-Learning
# ════════════════════════════════════════════════════════════════════════════════
class QLearningAgent(BaseComparisonAgent):
    def train(self):
        validation_curve = []
        is_train_success = False

        for episode in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1

                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                td_error = (
                    reward
                    + self.config["gamma"] * np.max(self.q_table[next_observation, :])
                    - self.q_table[observation, action]
                )
                self.q_table[observation, action] += self.config["alpha"] * td_error

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    _, avg_reward = self.validate()
                    validation_curve.append((self.time_steps, avg_reward))
                    if avg_reward == 1.0:
                        is_train_success = True

        return validation_curve, is_train_success


# ════════════════════════════════════════════════════════════════════════════════
# 2) SARSA
# ════════════════════════════════════════════════════════════════════════════════
class SARSAAgent(BaseComparisonAgent):
    def train(self):
        validation_curve = []
        is_train_success = False

        for episode in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            action = self.epsilon_greedy_action(observation)
            done = False

            while not done:
                self.time_steps += 1

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_action(next_observation)

                td_error = (
                    reward
                    + self.config["gamma"] * self.q_table[next_observation, next_action]
                    - self.q_table[observation, action]
                )
                self.q_table[observation, action] += self.config["alpha"] * td_error

                observation = next_observation
                action = next_action
                done = terminated or truncated

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    _, avg_reward = self.validate()
                    validation_curve.append((self.time_steps, avg_reward))
                    if avg_reward == 1.0:
                        is_train_success = True
                        # ✅ 수정 1: break 제거

        return validation_curve, is_train_success


# ════════════════════════════════════════════════════════════════════════════════
# 3) Expected SARSA
# ════════════════════════════════════════════════════════════════════════════════
class ExpectedSARSAAgent(BaseComparisonAgent):
    def _expected_q_value(self, observation):
        action_values = self.q_table[observation, :]
        num_actions = len(action_values)
        epsilon = self.config["epsilon"]

        max_value = np.max(action_values)
        greedy_actions = [a for a, v in enumerate(action_values) if v == max_value]
        num_greedy_actions = len(greedy_actions)

        action_probabilities = np.full(num_actions, epsilon / num_actions)
        for a in greedy_actions:
            action_probabilities[a] += (1.0 - epsilon) / num_greedy_actions

        return float(np.dot(action_probabilities, action_values))

    def train(self):
        validation_curve = []
        is_train_success = False

        for episode in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1

                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                td_error = (
                    reward
                    + self.config["gamma"] * self._expected_q_value(next_observation)
                    - self.q_table[observation, action]
                )
                self.q_table[observation, action] += self.config["alpha"] * td_error

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    _, avg_reward = self.validate()
                    validation_curve.append((self.time_steps, avg_reward))
                    if avg_reward == 1.0:
                        is_train_success = True
                        # ✅ 수정 1: break 제거

        return validation_curve, is_train_success


# ════════════════════════════════════════════════════════════════════════════════
# 4) Off-Policy SARSA (IS 가중치)
# ════════════════════════════════════════════════════════════════════════════════
class OffPolicySARSAAgent(BaseComparisonAgent):
    def _behavior_policy_probability(self, observation, action):
        action_values = self.q_table[observation, :]
        num_actions = len(action_values)
        epsilon = self.config["epsilon"]

        max_value = np.max(action_values)
        greedy_actions = [a for a, v in enumerate(action_values) if v == max_value]
        num_greedy_actions = len(greedy_actions)

        if action in greedy_actions:
            return (1.0 - epsilon) / num_greedy_actions + epsilon / num_actions
        else:
            return epsilon / num_actions

    def _target_policy_probability(self, observation, action):
        action_values = self.q_table[observation, :]
        max_value = np.max(action_values)
        greedy_actions = [a for a, v in enumerate(action_values) if v == max_value]

        if action in greedy_actions:
            return 1.0 / len(greedy_actions)
        else:
            return 0.0

    def train(self):
        validation_curve = []
        is_train_success = False

        for episode in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            action = self.epsilon_greedy_action(observation)
            done = False

            while not done:
                self.time_steps += 1

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_action(next_observation)

                behavior_prob = self._behavior_policy_probability(next_observation, next_action)
                target_prob = self._target_policy_probability(next_observation, next_action)
                importance_ratio = target_prob / behavior_prob if behavior_prob > 1e-10 else 0.0

                td_error = importance_ratio * (
                    reward
                    + self.config["gamma"] * self.q_table[next_observation, next_action]
                ) - self.q_table[observation, action]

                self.q_table[observation, action] += self.config["alpha"] * td_error

                observation = next_observation
                action = next_action
                done = terminated or truncated

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    _, avg_reward = self.validate()
                    validation_curve.append((self.time_steps, avg_reward))
                    if avg_reward == 1.0:
                        is_train_success = True
                        # ✅ 수정 1: break 제거

        return validation_curve, is_train_success


# ════════════════════════════════════════════════════════════════════════════════
# 실험 & 그래프
# ════════════════════════════════════════════════════════════════════════════════
def run_experiment(AgentClass, config, num_seeds):
    all_curves = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        random.seed(seed)
        agent = AgentClass(config=config)
        validation_curve, is_train_success = agent.train()
        all_curves.append(validation_curve)
        status = 'SUCCESS' if is_train_success else 'FAILED '
        print(f"  seed={seed}  {status:7s}  steps={agent.time_steps:,}")
    return all_curves


def simple_moving_average(values, window_size=7):
    if len(values) < window_size:
        return values
    cumsum = np.cumsum(values, dtype=float)
    sma = np.empty_like(values, dtype=float)
    for i in range(window_size - 1):
        sma[i] = cumsum[i] / (i + 1)
    sma[window_size - 1] = cumsum[window_size - 1] / window_size
    sma[window_size:] = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return sma


def interpolate_curves(all_curves, x_values):
    """
    all_curves  : [(time_step, reward), ...] 리스트의 리스트
    x_values    : 모든 알고리즘에 공통으로 사용할 x 좌표 배열  ← 수정 핵심
    """
    y_values_list = []
    for curve in all_curves:
        if not curve:
            continue
        xp = [point[0] for point in curve]
        fp = [point[1] for point in curve]
        # np.interp: x > xp[-1] 범위는 fp[-1](마지막 값)로 외삽
        # → 조기 수렴한 알고리즘도 이후 x 구간에서 마지막 성능값 유지
        y_values_list.append(np.interp(x_values, xp, fp, right=fp[-1]))
    return np.array(y_values_list)


def plot_comparison(results, config, save_path):
    style = {
        "Q-Learning":       {"color": "#1f77b4", "linestyle": "solid"},
        "SARSA":            {"color": "#ff7f0e", "linestyle": "dashed"},
        "Expected SARSA":   {"color": "#2ca02c", "linestyle": "dashdot"},
        "Off-Policy SARSA": {"color": "#d62728", "linestyle": "dotted"},
    }

    global_max_time_step = max(
        curve[-1][0]
        for curves in results.values()
        for curve in curves
        if curve
    )
    x_values = np.linspace(0, global_max_time_step, 300)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        "FrozenLake-v1 (4x4, non-slippery) - 4 TD Control Algorithms\n"
        f"alpha={config['alpha']}, gamma={config['gamma']}, epsilon={config['epsilon']}, "
        f"seeds={config['n_seeds']}",
        fontsize=12, y=1.01
    )

    subplot_titles = ["All Seeds + Mean", "Mean +/- Std (SMA)"]

    for ax_index, (ax, title) in enumerate(zip(axes, subplot_titles)):
        ax.set_title(title)
        ax.set_xlabel("Training Time Steps")
        ax.set_ylabel("Validation Avg. Episode Reward")

        for name, curves in results.items():
            s = style[name]
            y_values_array = interpolate_curves(curves, x_values)

            if ax_index == 0:
                for y_values in y_values_array:
                    ax.plot(
                        x_values, y_values,
                        color=s["color"], alpha=0.12, lw=0.9, zorder=1,
                    )
                mean_values = simple_moving_average(np.mean(y_values_array, axis=0), 12)
                ax.plot(
                    x_values, mean_values,
                    color=s["color"], lw=2.2, linestyle=s["linestyle"],
                    label=name, zorder=2,
                )
            else:
                mean_values = simple_moving_average(np.mean(y_values_array, axis=0), 15)
                std_values = simple_moving_average(np.std(y_values_array, axis=0), 15)
                ax.plot(
                    x_values, mean_values,
                    color=s["color"], lw=2.2, linestyle=s["linestyle"],
                    label=name, zorder=2,
                )
                ax.fill_between(
                    x_values, mean_values - std_values, mean_values + std_values,
                    color=s["color"], alpha=0.15, zorder=1,
                )

        ax.set_ylim(-0.05, 1.12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프 저장 --> {save_path}")
    plt.close()


def print_qtable(agent, name):
    print(f"\n[{name}] Final Q-Table")
    print("     LEFT    DOWN   RIGHT      UP")
    for idx, row in enumerate(agent.q_table):
        print(f" {idx:2d}: " + "  ".join(f"{value:6.3f}" for value in row))


def main():
    config = {
        "num_episodes": 5_000,
        "validation_time_steps_interval": 200,
        "validation_num_episodes": 3,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.1,
        "n_seeds": 10,
    }

    agents_definition = {
        "Q-Learning": QLearningAgent,
        "SARSA": SARSAAgent,
        "Expected SARSA": ExpectedSARSAAgent,
        "Off-Policy SARSA": OffPolicySARSAAgent,
    }

    results = {}
    for name, AgentClass in agents_definition.items():
        print(f"\n{'─' * 50}\n  {name}  ({config['n_seeds']} seeds)\n{'─' * 50}")
        results[name] = run_experiment(AgentClass, config, config["n_seeds"])

    plot_comparison(results, config, "./b_frozen_lake_sarsa_comparison_img.png")


if __name__ == "__main__":
    main()
