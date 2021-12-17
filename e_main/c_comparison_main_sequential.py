import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from a_configuration.parameter_comparison import ParameterComparison
from e_main.supports.main_preamble import *
from e_main.supports.learner_comparison import LearnerComparison
from g_utils.commons import print_comparison_basic_info, get_wandb_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameter_c = ParameterComparison()

n_agents = len(parameter_c.AGENT_PARAMETERS)

for agent_parameter in parameter_c.AGENT_PARAMETERS:
    del agent_parameter.MAX_TRAINING_STEPS
    del agent_parameter.N_ACTORS
    del agent_parameter.N_EPISODES_FOR_MEAN_CALCULATION
    del agent_parameter.N_TEST_EPISODES
    del agent_parameter.N_VECTORIZED_ENVS
    del agent_parameter.PROJECT_HOME
    del agent_parameter.TEST_INTERVAL_TRAINING_STEPS
    del agent_parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
    del agent_parameter.USE_WANDB
    del agent_parameter.WANDB_ENTITY
    del agent_parameter.MODEL_SAVE_DIR
    del agent_parameter.CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS


class ComparisonStat:
    def __init__(self):
        ###########################################
        ##### START: FOR WANDB GRAPHS LOGGING #####
        ###########################################
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
        #########################################
        ##### END: FOR WANDB GRAPHS LOGGING #####
        #########################################

    def save_fig(self):
        pass

    def save_csv(self):
        # 1
        column_names = [
            "MEAN_AVG_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_avg = pd.DataFrame(
            data=self.MEAN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MIN_AVG_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_min_avg = pd.DataFrame(
            data=self.MIN_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_AVG_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_max_avg = pd.DataFrame(
            data=self.MAX_test_episode_reward_avg_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 2
        column_names = [
            "MEAN_STD_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_std = pd.DataFrame(
            data=self.MEAN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_STD_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_min_std = pd.DataFrame(
            data=self.MIN_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_STD_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_max_std = pd.DataFrame(
            data=self.MAX_test_episode_reward_std_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        # 3
        column_names = [
            "MEAN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_mean_train = pd.DataFrame(
            data=self.MEAN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )

        column_names = [
            "MIN_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
        ]
        df_min_train = pd.DataFrame(
            data=self.MIN_mean_episode_reward_per_agent.T,
            index=self.test_training_steps_lst,
            columns=column_names
        )
        column_names = [
            "MAX_TRAIN_{0}".format(agent_idx) for agent_idx in range(len(parameter_c.AGENT_PARAMETERS))
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
                parameter_c.COMPARISON_RESULTS_SAVE_DIR, "{0}_runs_{1}_{2}.csv".format(
                    parameter_c.ENV_NAME, parameter_c.N_RUNS, formatted_now
                )
            ),
            float_format="%.3f"
        )


def main():
    print_comparison_basic_info(device, parameter_c)
    input("Press Enter to continue...")

    if parameter_c.USE_WANDB:
        wandb_obj = get_wandb_obj(parameter_c, comparison=True)
    else:
        wandb_obj = None

    obs_shape, n_actions = get_env_info(parameter_c)

    comparison_stat = ComparisonStat()

    print("\n########## LEARNING STARTED !!! ##########")
    for run in range(0, parameter_c.N_RUNS):
        print("\n" + ">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
        agents = []
        for agent_idx, _ in enumerate(parameter_c.AGENT_PARAMETERS):
            agent = get_agent(
                obs_shape=obs_shape, n_actions=n_actions, device=device,
                parameter=parameter_c.AGENT_PARAMETERS[agent_idx],
                max_training_steps=parameter_c.MAX_TRAINING_STEPS
            )
            agents.append(agent)

        learner_comparison = LearnerComparison(
            run=run, agents=agents, device=device, wandb_obj=wandb_obj,
            parameter_c=parameter_c, comparison_stat=comparison_stat
        )
        learner_comparison.train_comparison_loop()

    if parameter_c.USE_WANDB:
        wandb_obj.join()

    print_comparison_basic_info(device, parameter_c)

    comparison_stat.save_csv()
    comparison_stat.save_fig()



if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_c.N_ACTORS == 1 and parameter_c.N_VECTORIZED_ENVS == 1
    main()
