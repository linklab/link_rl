import os
import sys

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


class ComparisonStat:
    def __init__(self, n_agents):
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
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_avg_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_avg_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_avg_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
    
        # 2
        self.test_episode_reward_std_per_agent = np.zeros((
            parameter_c.N_RUNS,
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_test_episode_reward_std_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_test_episode_reward_std_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_test_episode_reward_std_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
    
        # 3
        self.mean_episode_reward_per_agent = np.zeros((
            parameter_c.N_RUNS,
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MIN_mean_episode_reward_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MEAN_mean_episode_reward_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        self.MAX_mean_episode_reward_per_agent = np.zeros((
            n_agents,
            int(parameter_c.MAX_TRAINING_STEPS // parameter_c.TEST_INTERVAL_TRAINING_STEPS)
        ))
        #########################################
        ##### END: FOR WANDB GRAPHS LOGGING #####
        #########################################


def main():
    print_comparison_basic_info(device, parameter_c)
    input("Press Enter to continue...")

    if parameter_c.USE_WANDB:
        wandb_obj = get_wandb_obj(parameter_c, comparison=True)
    else:
        wandb_obj = None

    obs_shape, n_actions = get_env_info(parameter_c)

    comparison_stat = ComparisonStat(len(parameter_c.AGENT_PARAMETERS))
        
    print("########## LEARNING STARTED !!! ##########")
    for run in range(0, parameter_c.N_RUNS):
        print(">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
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
        learner_comparison.train_loop()

    if parameter_c.USE_WANDB:
        wandb_obj.join()

    print_comparison_basic_info(device, parameter_c)


if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_c.N_ACTORS == 1 and parameter_c.N_VECTORIZED_ENVS == 1
    main()
