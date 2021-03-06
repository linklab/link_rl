# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import os, sys
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

print("PyTorch Version", torch.__version__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.e_utils.reward_changer import RewardChanger
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import load_model, map_range
from codes.e_utils.logger import get_logger
from codes.e_utils.names import EnvironmentName, AgentMode, RLAlgorithmName
from codes.e_utils.rl_utils import get_environment_input_output_info, MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX

my_logger = get_logger("openai_pendulum_ddpg")


def play_main(params, env):
    observation_shape, action_shape, num_outputs, action_n, action_min, action_max = get_environment_input_output_info(env)
    agent = rl_utils.get_rl_agent(
        observation_shape, action_shape, num_outputs, action_n, action_min, action_max, worker_id=-1, params=params, device=device
    )
    load_model(MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX, agent, inquery=False)
    agent.agent_mode = AgentMode.PLAY
    agent.model.eval()
    agent.test_model.load_state_dict(agent.model.state_dict())
    agent.test_model.eval()

    num_step = 0
    num_episode = 0

    while True:
        done = False
        episode_reward = 0

        if "Bullet" in params.ENVIRONMENT_ID.value:
            env.render()

        state = env.reset()

        num_episode += 1
        num_episode_step = 0

        agent_state = rl_utils.initial_agent_state()

        while not done:
            env.render()

            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)

            action, agent_state, = agent(state, agent_state)

            if isinstance(agent.model, ContinuousActionModel):
                action = map_range(
                    np.asarray(action),
                    np.ones_like(agent.action_min) * -1.0, np.ones_like(agent.action_max),
                    agent.action_min, agent.action_max
                )

            if action.ndim == 2:
                action = action[0]

            next_state, reward, done, info = env.step(action)

            if isinstance(env, RewardChanger):
                reward = env.reverse_reward(reward)

            episode_reward += reward

            state = next_state

            # if num_step % 1000 == 0:
            #     print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}".format(
            #         num_episode, num_episode_step, num_step
            #     ))

            if params.ENVIRONMENT_ID not in [
                EnvironmentName.PENDULUM_MATLAB_V0,
                EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
                EnvironmentName.REAL_DEVICE_RIP,
                EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
                EnvironmentName.QUANSER_SERVO_2
            ]:
                time.sleep(0.01)

        print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}, EPISODE DONE --> EPISODE REWARD: {3}".format(
            num_episode, num_episode_step, num_step, episode_reward
        ))

        time.sleep(0.1)


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters

    env = rl_utils.get_single_environment(params=params, mode=AgentMode.PLAY)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    play_main(params, env)
