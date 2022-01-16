from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

# 유니티 환경 경로
from a_configuration.a_parameter_template import parameter_list
from a_configuration.b_base.b_agents.agents import ParameterAgent
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from e_main.supports.learner import Learner
from e_main.supports.main_preamble import get_agent
from g_utils.commons import print_basic_info
from g_utils.types import AgentType, ModelType

env_name = "UnityEnvironment"

class Parameter3DBall():
    def __init__(self):
        self.ENV_NAME = "Unity3DBall"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 1024

class ParameterDqn(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.DQN

        self.LEARNING_RATE = 0.005

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TRAINING_STEP_PERCENT = 0.35

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10
        self.GAMMA = 0.99
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

class Parameter3DBallDqn(
    ParameterBase, Parameter3DBall, ParameterDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        Parameter3DBall.__init__(self)
        ParameterDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


parameter_3d_ball_dqn = Parameter3DBallDqn()
parameter_list.append(parameter_3d_ball_dqn)

parameter = parameter_3d_ball_dqn
parameter.USE_WANDB = False
parameter.PLAY_MODEL_FILE_NAME = ""

def main():
    # 유니티 환경 경로 설정 (file_name)
    u_env = UnityEnvironment(file_name=env_name, worker_id=1, no_graphics=False)
    env = UnityToGymWrapper(u_env)

    observation_space, action_space = env.observation_space, env.action_space

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, parameter=parameter
    )

    learner = Learner(agent=agent, queue=None, parameter=parameter)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)


if __name__ == "__main__":
    main()
