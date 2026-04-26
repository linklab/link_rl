# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import random
import cv2

import gymnasium as gym
import ale_py

import numpy as np


print("gym.__version__:", gym.__version__)

gym.register_envs(ale_py)

class CroppedAtariPreprocessing(gym.wrappers.AtariPreprocessing):
    """
    AtariPreprocessing을 상속해서 cv2.resize 직전에 crop 단계를 끼워 넣음.
    슬라이드 흐름:  (210, 160) → [crop] → (160, 160) → [resize] → (84, 84)
    """
    def __init__(self, env, top_crop=34, bottom_crop=16, **kwargs):
        super().__init__(env, **kwargs)
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def _get_obs(self):
        # 1) 두 프레임 max-pool (Atari 깜빡임 방지)
        if self.frame_skip > 1:
            np.maximum(
                self.obs_buffer[0], self.obs_buffer[1],
                out=self.obs_buffer[0]
            )

        # 2) 점수판/하단 여백 제거: (210, 160) → (160, 160)
        h = self.obs_buffer[0].shape[0]
        cropped = self.obs_buffer[0][self.top_crop:h - self.bottom_crop]

        # 3) 리사이즈: (160, 160) → (84, 84)
        obs = cv2.resize(
            cropped,
            self.screen_size,
            interpolation=cv2.INTER_AREA,
        )

        # 4) 정규화
        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        return obs

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")

env = CroppedAtariPreprocessing(
    env,
    noop_max=30,
    top_crop=34,  # 상단 점수판 영역
    bottom_crop=16,  # 하단 여백
    frame_skip=4,
    screen_size=(84, 84),
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=True
)

env = gym.wrappers.FrameStackObservation(env, stack_size=4)


ACTION_STRING_LIST = [" NOOP", " FIRE", "RIGHT", " LEFT"]

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

    # ---- 마지막 Observation 이미지 저장 ----
    # observation: (4, 84, 84), float32, [0.0, 1.0]
    import os
    from PIL import Image
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) float [0,1] -> uint8 [0,255]
    obs_uint8 = (observation * 255.0).clip(0, 255).astype(np.uint8)  # (4, 84, 84)

    # 2) 4개 프레임을 가로로 이어 붙여 하나의 이미지로 (84, 84*4)
    stacked_image = np.concatenate(list(obs_uint8), axis=1)
    save_path_stacked = os.path.join(script_dir, "last_observation_stacked.png")
    Image.fromarray(stacked_image, mode="L").save(save_path_stacked)

    # 3) (선택) 4개 프레임을 각각 별도 파일로도 저장
    for i, frame in enumerate(obs_uint8):
        save_path_i = os.path.join(script_dir, f"last_observation_frame_{i}.png")
        Image.fromarray(frame, mode="L").save(save_path_i)

    print(f"\n마지막 Observation 이미지 저장 완료")
    print(f"  원본 shape: {observation.shape}, dtype: {observation.dtype}")
    print(f"  - 4프레임 가로결합: {save_path_stacked}  (shape: {stacked_image.shape})")
    print(f"  - 개별 프레임 4장: last_observation_frame_0.png ~ _3.png")

if __name__ == "__main__":
    run_env()
