"""
c_frozen_lake_double_q_learning_comparison.py
────────────────────────────────────────────────────────────────────────────────
Q-Learning vs Double Q-Learning 성능 비교 (FrozenLake-v1, 4x4, non-slippery)
  1) Q-Learning        — max 편향(maximization bias) 존재
  2) Double Q-Learning — 두 개의 Q-Table로 max 편향 완화

gymnasium FrozenLake-v1 환경을 사용합니다.
b_frozen_lake_sarsa_comparison.py의 구조를 재사용합니다.
────────────────────────────────────────────────────────────────────────────────
"""

import random
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from a_frozen_lake_q_learning import QTableAgent, IS_SLIPPERY, MAP_NAME, DESC
from b_frozen_lake_sarsa_comparison import (
    BaseComparisonAgent, QLearningAgent,
    run_experiment, simple_moving_average, interpolate_curves, print_qtable,
)


# ════════════════════════════════════════════════════════════════════════════════
# Double Q-Learning
# ════════════════════════════════════════════════════════════════════════════════
class DoubleQLearningAgent(BaseComparisonAgent):
    def __init__(self, config):
        super().__init__(config)
        # Q-Table 두 개 사용
        self.q_table_1 = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.q_table_2 = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        # q_table은 두 테이블의 합으로 유지 (greedy_action, validate 등에서 사용)
        self.q_table = self.q_table_1 + self.q_table_2

    def _update_combined_q_table(self):
        np.add(self.q_table_1, self.q_table_2, out=self.q_table)

    def train(self):
        validation_curve = []
        is_train_success = False

        for episode in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1

                # 행동 선택: Q1 + Q2 기반 epsilon-greedy
                self._update_combined_q_table()
                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                # 50% 확률로 Q1 또는 Q2 업데이트
                if np.random.rand() < 0.5:
                    # Q1 업데이트: Q1(s,a) += alpha * (r + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))
                    best_action_by_q1 = np.argmax(self.q_table_1[next_observation, :])
                    td_error = (
                        reward
                        + self.config["gamma"] * self.q_table_2[next_observation, best_action_by_q1]
                        - self.q_table_1[observation, action]
                    )
                    self.q_table_1[observation, action] += self.config["alpha"] * td_error
                else:
                    # Q2 업데이트: Q2(s,a) += alpha * (r + gamma * Q1(s', argmax_a' Q2(s',a')) - Q2(s,a))
                    best_action_by_q2 = np.argmax(self.q_table_2[next_observation, :])
                    td_error = (
                        reward
                        + self.config["gamma"] * self.q_table_1[next_observation, best_action_by_q2]
                        - self.q_table_2[observation, action]
                    )
                    self.q_table_2[observation, action] += self.config["alpha"] * td_error

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    self._update_combined_q_table()
                    _, avg_reward = self.validate()
                    validation_curve.append((self.time_steps, avg_reward))
                    if avg_reward == 1.0:
                        is_train_success = True

        self._update_combined_q_table()
        return validation_curve, is_train_success


# ════════════════════════════════════════════════════════════════════════════════
# 그래프
# ════════════════════════════════════════════════════════════════════════════════
def plot_comparison(results, config, save_path):
    palette = {
        "Q-Learning":        "#1f77b4",
        "Double Q-Learning": "#d62728",
    }

    # 모든 알고리즘 전체 curves에서 global_max_time_step 계산
    global_max_time_step = max(
        curve[-1][0]
        for curves in results.values()
        for curve in curves
        if curve
    )
    x_values = np.linspace(0, global_max_time_step, 300)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        "FrozenLake-v1 (4x4, non-slippery) - Q-Learning vs Double Q-Learning\n"
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
            color = palette[name]
            y_values_array = interpolate_curves(curves, x_values)

            if ax_index == 0:
                for y_values in y_values_array:
                    ax.plot(x_values, y_values, color=color, alpha=0.12, lw=0.9)
                mean_values = simple_moving_average(np.mean(y_values_array, axis=0), 12)
                ax.plot(x_values, mean_values, color=color, lw=2.2, label=name)
            else:
                mean_values = simple_moving_average(np.mean(y_values_array, axis=0), 15)
                std_values = simple_moving_average(np.std(y_values_array, axis=0), 15)
                ax.plot(x_values, mean_values, color=color, lw=2.2, label=name)
                ax.fill_between(
                    x_values, mean_values - std_values, mean_values + std_values,
                    color=color, alpha=0.15
                )

        ax.set_ylim(-0.05, 1.12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프 저장 --> {save_path}")
    plt.close()


def main():
    config = {
        "num_episodes": 500,
        "validation_time_steps_interval": 30,
        "validation_num_episodes": 10,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.1,
        "n_seeds": 10,
    }

    agents_definition = {
        "Q-Learning": QLearningAgent,
        "Double Q-Learning": DoubleQLearningAgent,
    }

    results = {}
    for name, AgentClass in agents_definition.items():
        print(f"\n{'─' * 50}\n  {name}  ({config['n_seeds']} seeds)\n{'─' * 50}")
        results[name] = run_experiment(AgentClass, config, config["n_seeds"])

    plot_comparison(results, config, "./c_frozen_lake_double_q_learning_comparison_img.png")

    print("\n" + "═" * 50 + "\n  Final Q-Tables\n" + "═" * 50)
    for name, AgentClass in agents_definition.items():
        np.random.seed(config["n_seeds"] - 1)
        random.seed(config["n_seeds"] - 1)
        agent = AgentClass(config=config)
        agent.train()
        print_qtable(agent, name)


if __name__ == "__main__":
    main()