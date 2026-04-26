# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import random

import gymnasium as gym
import ale_py

import numpy as np

print("gym.__version__:", gym.__version__)

gym.register_envs(ale_py)

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")

ACTION_STRING_LIST = [" NOOP", " FIRE", "RIGHT", " LEFT", "RIGHTFIRE", "LEFTFIRE"]

print(env.observation_space) # Box(0, 255, (210, 160, 3), uint8)
print(env.action_space)      # Discrete(4)


class Dummy_Agent:
    def get_action(self, observation: np.ndarray) -> int:
        available_action_ids = [0, 1, 2, 3]
        action_id = random.choice(available_action_ids)
        return action_id


def run_env():
    print("START RUN!!!")
    agent = Dummy_Agent()
    observation, info = env.reset()

    done = False
    episode_step = 1
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        print(
            "[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4}, "
            "Reward: {5}, terminated: {6}, Truncated: {7}, Info: {8}".format(
                episode_step,
                str(observation.shape),
                action,
                ACTION_STRING_LIST[action],
                str(next_observation.shape),
                reward,
                terminated,
                truncated,
                info,
            )
        )
        observation = next_observation
        done = terminated or truncated
        episode_step += 1

    # 마지막 Observation 이미지를 스크립트와 동일한 폴더에 저장
    import os
    from PIL import Image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "last_observation.png")
    Image.fromarray(observation).save(save_path)
    print(f"\n마지막 Observation 이미지 저장 완료: {save_path}")
    print(f"  shape: {observation.shape}, dtype: {observation.dtype}")

if __name__ == "__main__":
    run_env()
