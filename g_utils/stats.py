import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


class ComparisonStat:
    def __init__(self, parameter_c):
        self.parameter_c = parameter_c

        self.test_training_steps_lst = []

        for step in range(
                parameter_c.TEST_INTERVAL_TRAINING_STEPS,
                parameter_c.MAX_TRAINING_STEPS,
                parameter_c.TEST_INTERVAL_TRAINING_STEPS,
        ):
            self.test_training_steps_lst.append(step)

        # 1
        self.test_episode_reward_avg_per_agent = np.zeros((
            parameter_c.N_RUNS,
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_avg_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_avg_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_avg_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

        # 2
        self.test_episode_reward_std_per_agent = np.zeros((
            parameter_c.N_RUNS,
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_std_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_std_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_std_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

        # 3
        self.mean_episode_reward_per_agent = np.zeros((
            parameter_c.N_RUNS,
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_mean_episode_reward_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_mean_episode_reward_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_mean_episode_reward_per_agent = np.zeros((
            len(parameter_c.AGENT_PARAMETERS),
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))

    def save_fig(self):
        # Test Average Episode Reward
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_avg_per_agent[0],
            label="Agent 0"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_avg_per_agent[1],
            label="Agent 1"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_avg_per_agent[2],
            label="Agent 2"
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_avg_per_agent[0],
            y2=self.MAX_test_episode_reward_avg_per_agent[0],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_avg_per_agent[1],
            y2=self.MAX_test_episode_reward_avg_per_agent[1],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_avg_per_agent[2],
            y2=self.MAX_test_episode_reward_avg_per_agent[2],
            alpha=0.2
        )

        plt.ylabel("Test average episode reward")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        now = time.time()
        formatted_now = time.strftime('%H_%M_%S', time.gmtime(now))
        plt.savefig(
            os.path.join(
                self.parameter_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs_{1}_{2}_avg.png".format(
                    self.parameter_c.ENV_NAME, self.parameter_c.N_RUNS, formatted_now
                )
            )
        )

        # Test Episode Reward Standard Deviation
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_std_per_agent[0],
            label="Agent 0"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_std_per_agent[1],
            label="Agent 1"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_test_episode_reward_std_per_agent[2],
            label="Agent 2"
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_std_per_agent[0],
            y2=self.MAX_test_episode_reward_std_per_agent[0],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_std_per_agent[1],
            y2=self.MAX_test_episode_reward_std_per_agent[1],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_test_episode_reward_std_per_agent[2],
            y2=self.MAX_test_episode_reward_std_per_agent[2],
            alpha=0.2
        )

        plt.ylabel("Episode reward standard deviation")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        now = time.time()
        formatted_now = time.strftime('%H_%M_%S', time.gmtime(now))
        plt.savefig(
            os.path.join(
                self.parameter_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs_{1}_{2}_std.png".format(
                    self.parameter_c.ENV_NAME, self.parameter_c.N_RUNS, formatted_now
                )
            )
        )

        # Training Average Episode Reward
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_mean_episode_reward_per_agent[0],
            label="Agent 0"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_mean_episode_reward_per_agent[1],
            label="Agent 1"
        )
        plt.plot(
            self.test_training_steps_lst,
            self.MEAN_mean_episode_reward_per_agent[2],
            label="Agent 2"
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_mean_episode_reward_per_agent[0],
            y2=self.MAX_mean_episode_reward_per_agent[0],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_mean_episode_reward_per_agent[1],
            y2=self.MAX_mean_episode_reward_per_agent[1],
            alpha=0.2
        )

        plt.fill_between(
            self.test_training_steps_lst,
            y1=self.MIN_mean_episode_reward_per_agent[2],
            y2=self.MAX_mean_episode_reward_per_agent[2],
            alpha=0.2
        )

        plt.ylabel("Training average episode reward")
        plt.xlabel("Training steps")
        plt.legend(loc="best", fancybox=True, framealpha=0.3)

        now = time.time()
        formatted_now = time.strftime('%H_%M_%S', time.gmtime(now))
        plt.savefig(
            os.path.join(
                self.parameter_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs_{1}_{2}_train_avg.png".format(
                    self.parameter_c.ENV_NAME, self.parameter_c.N_RUNS, formatted_now
                )
            )
        )

    def save_csv(self):
        # 1
        column_names = [
            "MEAN_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_avg = pd.DataFrame(
            data=self.MEAN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MIN_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_min_avg = pd.DataFrame(
            data=self.MIN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_AVG_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_max_avg = pd.DataFrame(
            data=self.MAX_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 2
        column_names = [
            "MEAN_STD_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_std = pd.DataFrame(
            data=self.MEAN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_STD_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_min_std = pd.DataFrame(
            data=self.MIN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_STD_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_max_std = pd.DataFrame(
            data=self.MAX_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 3
        column_names = [
            "MEAN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_train = pd.DataFrame(
            data=self.MEAN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
        ]
        df_min_train = pd.DataFrame(
            data=self.MIN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(self.parameter_c.AGENT_PARAMETERS))
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

        now = time.time()
        formatted_now = time.strftime('%H_%M_%S', time.gmtime(now))
        df_combined.to_csv(
            os.path.join(
                self.parameter_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs_{1}_{2}.csv".format(
                    self.parameter_c.ENV_NAME, self.parameter_c.N_RUNS, formatted_now
                )
            ),
            float_format="%.3f"
        )

