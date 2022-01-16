from a_configuration.a_parameter_template import parameter_3d_ball_ddpg
from e_main.supports.learner import Learner
from e_main.supports.main_preamble import get_agent
from g_utils.commons import get_env_info, print_basic_info


#parameter.py
parameter = parameter_3d_ball_ddpg
parameter.USE_WANDB = False
parameter.PLAY_MODEL_FILE_NAME = ""

# 유니티 환경 경로 설정 (file_name)


def main():
    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space, action_space, parameter)

    print("observation_space : ", observation_space)
    print("action_space : ", action_space)
    agent = get_agent(
        observation_space=observation_space, action_space=action_space, parameter=parameter
    )

    learner = Learner(agent=agent, queue=None, parameter=parameter)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)


if __name__ == "__main__":
    main()


