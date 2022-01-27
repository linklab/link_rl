import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


class ComparisonStat:
    def __init__(self, config_c):
        self.config_c = config_c

        self.test_training_steps_lst = []

        for step in range(
                config_c.TEST_INTERVAL_TRAINING_STEPS,
                config_c.MAX_TRAINING_STEPS,
                config_c.TEST_INTERVAL_TRAINING_STEPS,
        ):
            self.test_training_steps_lst.append(step)

        # 1
        self.test_episode_reward_avg_per_agent = np.zeros((
            config_c.N_RUNS,
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_avg_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_avg_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_avg_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

        # 2
        self.test_episode_reward_std_per_agent = np.zeros((
            config_c.N_RUNS,
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_std_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_std_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_std_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

        # 3
        self.mean_episode_reward_per_agent = np.zeros((
            config_c.N_RUNS,
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_mean_episode_reward_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_mean_episode_reward_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_mean_episode_reward_per_agent = np.zeros((
            len(config_c.AGENT_PARAMETERS),
            int(config_c.MAX_TRAINING_STEPS // config_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

    def save_fig(self, local_now):
        # Test Average Episode Reward
        plt.figure(figsize=(12, 5))

        for i in range(len(self.config_c.AGENT_PARAMETERS)):
            plt.plot(
                self.test_training_steps_lst,
                self.MEAN_test_episode_reward_avg_per_agent[i],
                label=self.config_c.AGENT_LABELS[i]
            )

            plt.fill_between(
                self.test_training_steps_lst,
                y1=self.MIN_test_episode_reward_avg_per_agent[i],
                y2=self.MAX_test_episode_reward_avg_per_agent[i],
                alpha=0.2
            )

        plt.ylabel("Test average episode reward")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        plt.savefig(
            os.path.join(
                self.config_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs{1}_{2}_{3}_{4}_avg.png".format(
                    self.config_c.ENV_NAME, self.config_c.N_RUNS, local_now.year, local_now.month, local_now.day
                )
            )
        )

        # Test Episode Reward Standard Deviation
        plt.figure(figsize=(12, 5))

        for i in range(len(self.config_c.AGENT_PARAMETERS)):
            plt.plot(
                self.test_training_steps_lst,
                self.MEAN_test_episode_reward_std_per_agent[i],
                label=self.config_c.AGENT_LABELS[i]
            )

            plt.fill_between(
                self.test_training_steps_lst,
                y1=self.MIN_test_episode_reward_std_per_agent[i],
                y2=self.MAX_test_episode_reward_std_per_agent[i],
                alpha=0.2
            )

        plt.ylabel("Episode reward standard deviation")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        plt.savefig(
            os.path.join(
                self.config_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs{1}_{2}_{3}_{4}_std.png".format(
                    self.config_c.ENV_NAME, self.config_c.N_RUNS, local_now.year, local_now.month, local_now.day
                )
            )
        )

        # Training Average Episode Reward
        plt.figure(figsize=(12, 5))

        for i in range(len(self.config_c.AGENT_PARAMETERS)):
            plt.plot(
                self.test_training_steps_lst,
                self.MEAN_mean_episode_reward_per_agent[i],
                label=self.config_c.AGENT_LABELS[i]
            )

            plt.fill_between(
                self.test_training_steps_lst,
                y1=self.MIN_mean_episode_reward_per_agent[i],
                y2=self.MAX_mean_episode_reward_per_agent[i],
                alpha=0.2
            )

        plt.ylabel("Training average episode reward")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        plt.savefig(
            os.path.join(
                self.config_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs{1}_{2}_{3}_{4}_train_avg.png".format(
                    self.config_c.ENV_NAME, self.config_c.N_RUNS, local_now.year, local_now.month, local_now.day
                )
            )
        )

    def save_csv(self, local_now):
        # 1
        column_names = [
            "MEAN_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_mean_avg = pd.DataFrame(
            data=self.MEAN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MIN_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_min_avg = pd.DataFrame(
            data=self.MIN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_max_avg = pd.DataFrame(
            data=self.MAX_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 2
        column_names = [
            "MEAN_STD_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_mean_std = pd.DataFrame(
            data=self.MEAN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_STD_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_min_std = pd.DataFrame(
            data=self.MIN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_STD_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_max_std = pd.DataFrame(
            data=self.MAX_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 3
        column_names = [
            "MEAN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_mean_train = pd.DataFrame(
            data=self.MEAN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_min_train = pd.DataFrame(
            data=self.MIN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.config_c.AGENT_PARAMETERS))
        ]
        df_max_train = pd.DataFrame(
            data=self.MAX_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        df_combined = pd.concat([
            df_mean_avg, df_min_avg, df_max_avg,
            df_mean_std, df_min_std, df_max_std,
            df_mean_train, df_min_train, df_max_train,
        ], axis=1)

        df_combined.to_csv(
            os.path.join(
                self.config_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs{1}_{2}_{3}_{4}.csv".format(
                    self.config_c.ENV_NAME, self.config_c.N_RUNS, local_now.year, local_now.month, local_now.day
                )
            ),
            float_format="%.3f"
        )
